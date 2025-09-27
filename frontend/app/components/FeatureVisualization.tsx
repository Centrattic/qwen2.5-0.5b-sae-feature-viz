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
        <div className="space-y-6">
            {/* Feature Activation Visualization */}
            <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                    Feature {feature} Activations
                </h3>
                <div className="feature-visualization p-4 rounded-lg">
                    <div className="flex flex-wrap gap-2">
                        {data.activations.map((activation, index) => (
                            <div
                                key={index}
                                className="token-highlight px-3 py-2 rounded-md text-sm font-mono"
                                style={{
                                    backgroundColor: getActivationColor(activation),
                                    transform: `scale(${getActivationSize(activation)})`,
                                    color: Math.abs(activation) > 0.5 ? 'white' : 'black'
                                }}
                            >
                                <div className="text-xs">
                                    {data.tokens[index] || `token_${index}`}
                                </div>
                                <div className="text-xs font-bold">
                                    {activation.toFixed(3)}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Top Features for Each Token */}
            <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                    Top Features by Token
                </h3>
                <div className="space-y-2">
                    {data.topFeatures.map((tokenFeatures, tokenIndex) => (
                        <div key={tokenIndex} className="bg-gray-50 p-3 rounded-lg">
                            <div className="text-sm font-medium text-gray-700 mb-2">
                                Token {tokenIndex}: {data.tokens[tokenIndex] || `token_${tokenIndex}`}
                            </div>
                            <div className="flex flex-wrap gap-2">
                                {tokenFeatures.map((feature, featureIndex) => (
                                    <div
                                        key={featureIndex}
                                        className="px-2 py-1 bg-white rounded text-xs"
                                    >
                                        <span className="font-mono">F{feature.feature_idx}</span>
                                        <span className="ml-1 text-gray-600">
                                            {feature.activation.toFixed(3)}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Statistics */}
            <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                    Feature Statistics
                </h3>
                <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-50 p-3 rounded-lg">
                        <div className="text-sm text-gray-600">Max Activation</div>
                        <div className="text-lg font-semibold">
                            {Math.max(...data.activations).toFixed(3)}
                        </div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg">
                        <div className="text-sm text-gray-600">Min Activation</div>
                        <div className="text-lg font-semibold">
                            {Math.min(...data.activations).toFixed(3)}
                        </div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg">
                        <div className="text-sm text-gray-600">Mean Activation</div>
                        <div className="text-lg font-semibold">
                            {(data.activations.reduce((a, b) => a + b, 0) / data.activations.length).toFixed(3)}
                        </div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg">
                        <div className="text-sm text-gray-600">Active Tokens</div>
                        <div className="text-lg font-semibold">
                            {data.activations.filter(a => Math.abs(a) > 0.1).length}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
