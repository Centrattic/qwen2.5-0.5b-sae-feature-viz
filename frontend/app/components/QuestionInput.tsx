'use client'

import { useState } from 'react'
import { api, QuestionRequest } from '../lib/api'

interface QuestionInputProps {
    onQuestionSubmit: (question: string) => void
    selectedModel: string
}

export default function QuestionInput({ onQuestionSubmit, selectedModel }: QuestionInputProps) {
    const [question, setQuestion] = useState('')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!question.trim() || !selectedModel) return

        setLoading(true)
        setError('')

        try {
            const request: QuestionRequest = {
                question: question.trim(),
                model_name: selectedModel
            }

            const response = await api.processQuestion(request)

            if (response.success) {
                onQuestionSubmit(question.trim())
                setQuestion('')
            } else {
                setError('Failed to process question')
            }
        } catch (err) {
            setError('Error processing question')
            console.error('Error:', err)
        } finally {
            setLoading(false)
        }
    }

    return (
        <form onSubmit={handleSubmit} className="space-y-3">
            <div>
                <textarea
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Enter your question here..."
                    className="w-full px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                    rows={2}
                    disabled={loading}
                />
            </div>

            {error && (
                <div className="text-xs text-red-600 bg-red-50 p-1 rounded">
                    {error}
                </div>
            )}

            <button
                type="submit"
                disabled={loading || !question.trim() || !selectedModel}
                className="w-full px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
                {loading ? 'Processing...' : 'Submit Question'}
            </button>
        </form>
    )
}
