'use client'

interface ModelSelectorProps {
    models: string[]
    selectedModel: string
    onModelChange: (model: string) => void
}

export default function ModelSelector({ models, selectedModel, onModelChange }: ModelSelectorProps) {
    const getModelDisplayName = (model: string) => {
        if (model.includes('bad-medical-advice')) return 'Bad Medical Advice'
        if (model.includes('extreme-sports')) return 'Extreme Sports'
        if (model.includes('risky-financial-advice')) return 'Risky Financial Advice'
        if (model.includes('Qwen2.5-0.5B-Instruct')) return 'Base Qwen2.5-0.5B'
        return model
    }

    return (
        <div className="space-y-2">
            {models.map((model) => (
                <label key={model} className="flex items-center space-x-3 cursor-pointer">
                    <input
                        type="radio"
                        name="model"
                        value={model}
                        checked={selectedModel === model}
                        onChange={(e) => onModelChange(e.target.value)}
                        className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300"
                    />
                    <span className="text-sm text-gray-700">
                        {getModelDisplayName(model)}
                    </span>
                </label>
            ))}
        </div>
    )
}
