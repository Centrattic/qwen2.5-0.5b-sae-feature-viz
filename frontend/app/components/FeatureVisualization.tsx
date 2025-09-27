'use client'

import { useState, useEffect } from 'react'
import { api, FeatureRequest, TopFeaturesRequest } from '../lib/api'

interface FeatureVisualizationProps {
    model: string
    feature: number
    question: string
}

interface VisualizationData {
    activations: number[]
    tokens: string[]
    topFeatures: Array<Array<{
        feature_idx: number
        activation: number
    }>>
}

export default function FeatureVisualization({ model, feature, question }: FeatureVisualizationProps) {
    const [data, setData] = useState<VisualizationData | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    useEffect(() => {
        if (model && question) {
            loadVisualizationData()
        }
    }, [model, question, feature])

    const loadVisualizationData = async () => {
        setLoading(true)
        setError('')

        try {
            // First, process the question to get the question hash
            const questionResponse = await api.processQuestion({
                question,
                model_name: model
            })

            if (!questionResponse.success) {
                throw new Error('Failed to process question')
            }

            // Get feature activations
            const featureRequest: FeatureRequest = {
                model_name: model,
                question_hash: questionResponse.question_hash,
                feature_idx: feature
            }

            const [featureResponse, topFeaturesResponse] = await Promise.all([
                api.getFeatureActivations(featureRequest),
                api.getTopFeatures({
                    model_name: model,
                    question_hash: questionResponse.question_hash,
                    top_k: 5
                })
            ])

            if (featureResponse.success && topFeaturesResponse.success) {
                setData({
                    activations: featureResponse.activations,
                    tokens: featureResponse.tokens,
                    topFeatures: topFeaturesResponse.top_features
                })
            } else {
                throw new Error('Failed to load visualization data')
            }
        } catch (err) {
            setError('Error loading visualization data')
            console.error('Error:', err)
        } finally {
            setLoading(false)
        }
    }

    const getActivationColor = (activation: number) => {
        const intensity = Math.min(Math.abs(activation) * 2, 1)
        const hue = activation > 0 ? 120 : 0 // Green for positive, red for negative
        return `hsl(${hue}, 70%, ${50 + intensity * 30}%)`
    }

    const getActivationSize = (activation: number) => {
        return Math.max(0.5, Math.min(2, Math.abs(activation) * 3))
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto"></div>
                    <p className="mt-2 text-gray-600">Loading visualization...</p>
                </div>
            </div>
        )
    }

    if (error) {
        return (
            <div className="text-center py-8">
                <div className="text-red-600 mb-2">Error</div>
                <p className="text-gray-600">{error}</p>
                <button
                    onClick={loadVisualizationData}
                    className="mt-4 px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700"
                >
                    Retry
                </button>
            </div>
        )
    }

    if (!data) {
        return (
            <div className="text-center py-8 text-gray-500">
                Select a model and question to view feature visualization
            </div>
        )
    }

    return (
        <div className="h-full flex flex-col bg-white">
            {/* Top Controls */}
            <div className="border-b border-gray-200 p-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                        <button className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded border">
                            Stacked
                        </button>
                        <button className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded border border-blue-300">
                            Snippet
                        </button>
                        <button className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded border">
                            Full
                        </button>
                    </div>
                    <div className="flex items-center space-x-2">
                        <button className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded border">
                            Show Raw Tokens
                        </button>
                        <button className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded border">
                            Show Formatted
                        </button>
                        <button className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded border">
                            Show Breaks
                        </button>
                        <button className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded border">
                            Hide Breaks
                        </button>
                    </div>
                </div>
            </div>

            {/* Main Visualization Area */}
            <div className="flex-1 overflow-y-auto p-6">
                <div className="space-y-4">
                    {/* Response snippets with highlighted tokens */}
                    {data.activations.map((activation, index) => {
                        const isHighActivation = Math.abs(activation) > 0.3
                        const activationColor = activation > 0 ? '#10b981' : '#ef4444'
                        const intensity = Math.min(Math.abs(activation) * 2, 1)

                        return (
                            <div key={index} className="border border-gray-200 rounded-lg p-4 bg-gray-50">
                                <div className="flex items-start justify-between mb-2">
                                    <div className="flex items-center space-x-2">
                                        <span className="text-sm font-medium text-gray-700">
                                            {data.tokens[index] || `token_${index}`}
                                        </span>
                                        <span className="text-sm font-mono text-gray-500">
                                            {activation.toFixed(3)}
                                        </span>
                                    </div>
                                    <div className="text-xs text-gray-500">
                                        Token {index}
                                    </div>
                                </div>

                                <div className="relative">
                                    <div
                                        className="inline-block px-2 py-1 rounded text-sm font-medium"
                                        style={{
                                            backgroundColor: isHighActivation ? activationColor : '#f3f4f6',
                                            color: isHighActivation ? 'white' : '#374151',
                                            opacity: isHighActivation ? 0.8 + (intensity * 0.2) : 0.6
                                        }}
                                    >
                                        {data.tokens[index] || `token_${index}`}
                                        <span className="ml-2 text-xs">
                                            {activation.toFixed(3)}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        )
                    })}
                </div>
            </div>

            {/* Bottom Panel with Statistics */}
            <div className="border-t border-gray-200 p-4 bg-gray-50">
                <div className="grid grid-cols-4 gap-4">
                    <div className="text-center">
                        <div className="text-xs text-gray-500 mb-1">Max Activation</div>
                        <div className="text-sm font-semibold text-gray-900">
                            {Math.max(...data.activations).toFixed(3)}
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="text-xs text-gray-500 mb-1">Min Activation</div>
                        <div className="text-sm font-semibold text-gray-900">
                            {Math.min(...data.activations).toFixed(3)}
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="text-xs text-gray-500 mb-1">Mean Activation</div>
                        <div className="text-sm font-semibold text-gray-900">
                            {(data.activations.reduce((a, b) => a + b, 0) / data.activations.length).toFixed(3)}
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="text-xs text-gray-500 mb-1">Active Tokens</div>
                        <div className="text-sm font-semibold text-gray-900">
                            {data.activations.filter(a => Math.abs(a) > 0.1).length}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
