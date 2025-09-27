'use client'

import { useState } from 'react'

interface FeatureSelectorProps {
    selectedFeature: number
    onFeatureChange: (feature: number) => void
}

export default function FeatureSelector({ selectedFeature, onFeatureChange }: FeatureSelectorProps) {
    const [inputValue, setInputValue] = useState(selectedFeature.toString())

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = e.target.value
        setInputValue(value)

        const numValue = parseInt(value)
        if (!isNaN(numValue) && numValue >= 0 && numValue < 28672) {
            onFeatureChange(numValue)
        }
    }

    const handleRandomFeature = () => {
        const randomFeature = Math.floor(Math.random() * 28672)
        setInputValue(randomFeature.toString())
        onFeatureChange(randomFeature)
    }

    return (
        <div className="space-y-4">
            <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                    SAE Feature Index (0-28671)
                </label>
                <div className="flex space-x-2">
                    <input
                        type="number"
                        min="0"
                        max="28671"
                        value={inputValue}
                        onChange={handleInputChange}
                        className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                        placeholder="Enter feature index"
                    />
                    <button
                        onClick={handleRandomFeature}
                        className="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500"
                    >
                        Random
                    </button>
                </div>
            </div>

            <div className="text-xs text-gray-500">
                Selected: Feature {selectedFeature}
            </div>
        </div>
    )
}
