from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime

from src.config import FRIEND_ID, FRIEND_NAME, RIYA_NAME, RIYA_ID, OLD_RESULTS_FOLDER


def aggregate_activations(  # ToDo: is there a way to average across the top few?
    acts: torch.Tensor,
    method: str,
    top_k:
    int = 15,  # Success depends on whether magnitude of padding tokens is large 😭 idk should research this
) -> torch.Tensor:

    if acts.ndim == 2:
        acts = acts.unsqueeze(0)  # make it (1, seq_len, H)

    B, S, H = acts.shape

    if method == "mean_top_k":
        norms = acts.norm(dim=2)  # (B,S)
        # get top_k indices (or all if seq_len < top_k)
        k = min(top_k, S)
        topk = norms.topk(k, dim=1).indices
        batch_idxs = torch.arange(B, device=acts.device)[:, None]
        selected = acts[batch_idxs, topk]  # (B, K, H)
        agg = selected.mean(dim=1)  # (B, H)
    elif method == "max":
        agg = acts.max(dim=1).values
    elif method == "last":
        agg = acts[:, -1, :]
    elif method == "mean":
        agg = acts.mean(dim=1)
    else:
        raise ValueError(f"Unknown method: {method}")

    return agg.squeeze(0) if B == 1 else agg


# GPT method to fix datetime parsing. TBH i should just be able to save it better myself.
def _parse_and_format_ts(ts: str) -> str:
    # Trim fractional seconds to 6 digits for fromisoformat
    if '+' in ts or '-' in ts[19:]:
        # split out timezone part
        for sep in ('+', '-'):
            if sep in ts[19:]:
                main, tz = ts.split(sep, 1)
                tz = sep + tz
                break
    else:
        main, tz = ts, ''
    if '.' in main:
        date, frac = main.split('.')
        frac6 = (frac + '000000')[:6]
        main6 = f"{date}.{frac6}"
    else:
        main6 = main
    dt = datetime.fromisoformat(main6 + tz)
    return dt.strftime("%m-%d-%y-%H-%M-%S")


def find_topk_train_samples(
    cache,  # A FinalLayerActivationCache instance
    input_acts: np.ndarray,  # (hidden_size,)
    k: int,
    author_id: str,
    agg_method: str,
    device: str = "cuda",
) -> Tuple[List[str], List[str]]:
    """
    Returns:
      entries: ["{author} {timestamp}: {content}", ...] length k
      prompts: [content, ...] length k
    """
    # Load reverse maps
    author_map = json.loads(
        cache.paths.author_map.read_text())  # {hash:author}
    content_map = json.loads(
        cache.paths.content_map.read_text())  # {hash:[ts,content]}

    author_hashes = [h for h, a in author_map.items() if a == author_id]
    if not author_hashes:
        return [], []

    rows = [cache._index[h] for h in author_hashes if h in cache._index]
    if not rows:
        return [], []

    # Load all activations + hashes
    acts_mm = cache._open_act(writable=False)
    subset_np = acts_mm[rows]

    # activations[0] is (1, prompt_len, hidden_size)
    # activations[1] is (1, prompt_len+1, hidden_size)
    # etc.

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    acts_t = torch.from_numpy(subset_np).to(device)

    vecs = aggregate_activations(acts_t, method=agg_method)
    query = input_acts.to(device).float() if isinstance(input_acts, torch.Tensor) \
            else torch.from_numpy(input_acts).to(device).float()

    # Cosine similarity
    sims = F.cosine_similarity(vecs, query.unsqueeze(0), dim=-1)

    # Top-k
    topk_vals, topk_idxs = sims.topk(k, largest=True)
    selected_rows = [rows[i] for i in topk_idxs.cpu().tolist()]

    hashes_mm = np.memmap(cache.paths.hash,
                          dtype=cache.HASH_DTYPE,
                          mode="r",
                          shape=(cache.rows, ))
    entries, prompts = [], []
    for r in selected_rows:
        h = hashes_mm[r].decode()
        ts, content = content_map.get(h, ("<unk>", ""))
        # nice timestamp
        dt = _parse_and_format_ts(ts)
        # pretty author name
        author = FRIEND_NAME if author_id == str(FRIEND_ID) \
                 else RIYA_NAME if author_id == str(RIYA_ID) else "UNKNOWN"
        entries.append(f"{author} {dt}: {content}")
        prompts.append(content)

    return entries, prompts


def make_hash(ts: str, content: str) -> str:
    norm_ts = normalize_timestamp(ts)
    norm_ct = normalize_content(content)
    return hashlib.sha1(f"{norm_ts}|{norm_ct}".encode()).hexdigest()


def normalize_timestamp(ts: str) -> str:
    """Return original string if parsing fails (still deterministic)."""
    return ts.strip()


def normalize_content(text: str) -> str:
    """Trim + collapse internal whitespace"""
    return " ".join(text.strip().split())


def pad_or_truncate(arr: torch.Tensor, seq_len: int) -> torch.Tensor:
    if arr.size(1) == seq_len:
        return arr
    if arr.size(1) > seq_len:
        return arr[:, :seq_len]
    # pad
    pad = seq_len - arr.size(1)
    return torch.nn.functional.pad(arr, (0, 0, 0, pad))


@dataclass
class CachePaths:
    act: Path
    hash: Path
    shape: Path
    author_map: Path
    meta: Path
    content_map: Path


class SingleLayerActivationCache:
    HASH_DTYPE = np.dtype("S40")

    def __init__(
        self,
        hidden_size: int,
        max_seq_len: int,
        base_dir: Path = Path(f"./{OLD_RESULTS_FOLDER}/activation_cache"),
        prefix: str = "final",
    ):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.prefix = prefix
        self._content_map: Dict[str, Tuple[str, str]] = {}
        self.paths = CachePaths(
            act=self.base_dir / f"{prefix}.acts.mmap",
            hash=self.base_dir / f"{prefix}.hash.mmap",
            shape=self.base_dir / f"{prefix}.shape.json",
            author_map=self.base_dir / f"{prefix}.author_index.json",
            meta=self.base_dir / f"{prefix}.meta.json",
            content_map=self.base_dir / f"{prefix}.content_map.json")
        self._row_bytes = self.max_seq_len * self.hidden_size * np.dtype(
            np.float16).itemsize
        self._index: Dict[str, int] = {}
        self._author_index: Dict[str, str] = {}
        self._load_indices()

    @property
    def rows(self) -> int:
        return self._rows_from_shape()

    def has(self, h: str) -> bool:
        return h in self._index

    def add_batch(
            self,
            hashes: Sequence[str],
            authors: Sequence[str],
            timestamps: List[str],
            contents: List[str],
            activations: torch.Tensor,  # (B, seq_len, hidden)
    ) -> None:
        assert activations.dtype in (torch.float16, torch.bfloat16,
                                     torch.float32)
        acts = activations.detach().cpu().to(torch.float16).numpy()
        B, S, H = acts.shape
        assert H == self.hidden_size
        # resize files
        old = self.rows
        new = old + B
        act_mm = self._grow_act(old, new)
        hash_mm = self._grow_hash(old, new)
        # write
        act_mm[old:new] = acts
        hash_mm[old:new] = np.array(hashes, dtype=self.HASH_DTYPE)
        act_mm.flush()
        hash_mm.flush()
        # update shape + indices
        self._write_shape(new)
        for i, h in enumerate(hashes):
            self._index[h] = old + i
            self._author_index[h] = authors[i]
            self._content_map[h] = (timestamps[i], contents[i])
        self._flush_author_index()

    def get_by_hashes(self, hashes: Sequence[str]) -> np.ndarray:
        if not hashes:
            return np.empty((0, self.max_seq_len, self.hidden_size),
                            dtype=np.float16)
        act_mm = self._open_act(writable=False)
        rows = [self._index[h] for h in hashes if h in self._index]
        return act_mm[rows]

    def hashes_for_author(self, author_id: str) -> List[str]:
        return [h for h, a in self._author_index.items() if a == author_id]

    def _load_indices(self):
        if self.paths.hash.exists():
            hashes = np.memmap(self.paths.hash,
                               dtype=self.HASH_DTYPE,
                               mode="r")
            self._index = {h.decode(): i for i, h in enumerate(hashes)}
        if self.paths.author_map.exists():
            self._author_index = json.loads(self.paths.author_map.read_text())
        if self.paths.content_map.exists():
            self._content_map = json.loads(self.paths.content_map.read_text())

    def _flush_author_index(self):
        self.paths.author_map.write_text(json.dumps(self._author_index))
        self.paths.content_map.write_text(json.dumps(self._content_map))

    def _rows_from_shape(self) -> int:
        if not self.paths.shape.exists():
            return 0
        return json.loads(self.paths.shape.read_text()).get("rows", 0)

    def _write_shape(self, rows: int):
        self.paths.shape.write_text(json.dumps({"rows": rows}))
        self.paths.meta.write_text(
            json.dumps({
                "rows": rows,
                "hidden_size": self.hidden_size,
                "max_seq_len": self.max_seq_len,
            }))

    def _open_act(self, writable: bool):
        rows = self.rows
        if rows == 0:
            shape = (0, self.max_seq_len, self.hidden_size)
            return np.memmap(self.paths.act,
                             dtype=np.float16,
                             mode="w+" if writable else "w+",
                             shape=shape)
        mode = "r+" if writable else "r"
        return np.memmap(self.paths.act,
                         dtype=np.float16,
                         mode=mode,
                         shape=(rows, self.max_seq_len, self.hidden_size))

    def _grow_act(self, old_rows: int, new_rows: int):
        shape = (new_rows, self.max_seq_len, self.hidden_size)
        if not self.paths.act.exists():
            return np.memmap(self.paths.act,
                             dtype=np.float16,
                             mode="w+",
                             shape=shape)
        old_mm = np.memmap(self.paths.act,
                           dtype=np.float16,
                           mode="r",
                           shape=(old_rows, self.max_seq_len,
                                  self.hidden_size))
        tmp_path = self.paths.act.with_suffix(".tmp")
        tmp = np.memmap(tmp_path, dtype=np.float16, mode="w+", shape=shape)
        tmp[:old_rows] = old_mm[:]
        self.paths.act.unlink()
        tmp.flush()
        tmp._mmap.close()
        tmp_path.rename(self.paths.act)
        return np.memmap(self.paths.act,
                         dtype=np.float16,
                         mode="r+",
                         shape=shape)

    def _grow_hash(self, old_rows: int, new_rows: int):
        shape = (new_rows, )
        if not self.paths.hash.exists():
            return np.memmap(self.paths.hash,
                             dtype=self.HASH_DTYPE,
                             mode="w+",
                             shape=shape)
        old_mm = np.memmap(self.paths.hash,
                           dtype=self.HASH_DTYPE,
                           mode="r",
                           shape=(old_rows, ))
        tmp_path = self.paths.hash.with_suffix(".tmp")
        tmp = np.memmap(tmp_path,
                        dtype=self.HASH_DTYPE,
                        mode="w+",
                        shape=shape)
        tmp[:old_rows] = old_mm[:]
        self.paths.hash.unlink()
        tmp.flush()
        tmp._mmap.close()
        tmp_path.rename(self.paths.hash)
        return np.memmap(self.paths.hash,
                         dtype=self.HASH_DTYPE,
                         mode="r+",
                         shape=shape)