'use client'

import { useState, useEffect } from 'react'
import ModelSelector from './components/ModelSelector'
import FeatureSelector from './components/FeatureSelector'
import QuestionList from './components/QuestionList'
import FeatureVisualization from './components/FeatureVisualization'
import QuestionInput from './components/QuestionInput'
import { API_BASE_URL } from './lib/api'

export default function Home() {
    const [selectedModel, setSelectedModel] = useState<string>('')
    const [selectedFeature, setSelectedFeature] = useState<number>(0)
    const [selectedQuestion, setSelectedQuestion] = useState<string>('')
    const [models, setModels] = useState<string[]>([])
    const [questions, setQuestions] = useState<string[]>([])
    const [loading, setLoading] = useState(false)

    useEffect(() => {
        loadInitialData()
    }, [])

    const loadInitialData = async () => {
        setLoading(true)
        try {
            const [modelsRes, questionsRes] = await Promise.all([
                fetch(`${API_BASE_URL}/models`),
                fetch(`${API_BASE_URL}/questions`)
            ])

            const modelsData = await modelsRes.json()
            const questionsData = await questionsRes.json()

            setModels(modelsData.models)
            setQuestions(questionsData.questions)

            if (modelsData.models.length > 0) {
                setSelectedModel(modelsData.models[0])
            }
            if (questionsData.questions.length > 0) {
                setSelectedQuestion(questionsData.questions[0])
            }
        } catch (error) {
            console.error('Error loading initial data:', error)
        } finally {
            setLoading(false)
        }
    }

    const handleQuestionSubmit = (newQuestion: string) => {
        setQuestions(prev => [...prev, newQuestion])
        setSelectedQuestion(newQuestion)
    }

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-gray-50">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto"></div>
                    <p className="mt-4 text-gray-600">Loading SAE Feature Visualization...</p>
                </div>
            </div>
        )
    }

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Top Navigation Bar */}
            <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
                <div className="max-w-full mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex items-center justify-between h-16">
                        <div className="flex items-center space-x-4">
                            <button className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded border">
                                PREV
                            </button>
                            <button className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded border">
                                NEXT
                            </button>
                        </div>

                        <div className="flex items-center space-x-4">
                            <select
                                value={selectedModel}
                                onChange={(e) => setSelectedModel(e.target.value)}
                                className="px-3 py-1 text-sm border border-gray-300 rounded bg-white"
                            >
                                {models.map(model => (
                                    <option key={model} value={model}>
                                        {model.includes('bad-medical-advice') ? 'Bad Medical Advice' :
                                            model.includes('extreme-sports') ? 'Extreme Sports' :
                                                model.includes('risky-financial-advice') ? 'Risky Financial Advice' :
                                                    model.includes('Qwen2.5-0.5B-Instruct') ? 'Base Qwen2.5-0.5B' : model}
                                    </option>
                                ))}
                            </select>

                            <span className="text-sm text-gray-500">8-RES-JH</span>

                            <div className="flex items-center space-x-2">
                                <input
                                    type="number"
                                    value={selectedFeature}
                                    onChange={(e) => setSelectedFeature(parseInt(e.target.value) || 0)}
                                    className="w-20 px-2 py-1 text-sm border border-gray-300 rounded"
                                    min="0"
                                    max="28671"
                                />
                                <button className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700">
                                    GO
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </header>

            <div className="flex h-screen">
                {/* Left Panel */}
                <div className="w-80 bg-white border-r border-gray-200 overflow-y-auto">
                    <div className="p-6 space-y-6">
                        {/* Explanations Section */}
                        <div>
                            <h3 className="text-sm font-semibold text-gray-900 mb-3">EXPLANATIONS</h3>
                            <div className="space-y-3">
                                <div className="text-sm text-gray-700">
                                    Feature {selectedFeature}: Activations for specific linguistic patterns and concepts.
                                </div>
                                <div className="text-xs text-gray-500">
                                    qwen2.5-0.5b-sae-feature-viz
                                </div>
                                <button className="w-full px-3 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-700">
                                    NEW AUTO-INTERP
                                </button>
                            </div>
                        </div>

                        {/* Configuration Section */}
                        <div>
                            <h3 className="text-sm font-semibold text-gray-900 mb-3">CONFIGURATION</h3>
                            <div className="space-y-2 text-xs text-gray-600">
                                <div>Hook Name: blocks.8.ln2.hook_normalized</div>
                                <div>Features: 28,672</div>
                                <div>Architecture: gated</div>
                                <div>Dataset: Qwen2.5-0.5B responses</div>
                                <div>Data Type: float16</div>
                                <div>Hook Layer: 8</div>
                                <div>Context Size: 512</div>
                                <div>Activation Function: relu</div>
                                <button className="text-blue-600 hover:text-blue-700">Show All</button>
                            </div>
                        </div>

                        {/* Questions Section */}
                        <div>
                            <h3 className="text-sm font-semibold text-gray-900 mb-3">QUESTIONS</h3>
                            <QuestionList
                                questions={questions}
                                selectedQuestion={selectedQuestion}
                                onQuestionSelect={setSelectedQuestion}
                            />
                        </div>

                        {/* Add Question Section */}
                        <div>
                            <h3 className="text-sm font-semibold text-gray-900 mb-3">ADD QUESTION</h3>
                            <QuestionInput
                                onQuestionSubmit={handleQuestionSubmit}
                                selectedModel={selectedModel}
                            />
                        </div>
                    </div>
                </div>

                {/* Main Content Area */}
                <div className="flex-1 flex flex-col">
                    {/* Feature Visualization */}
                    <div className="flex-1 overflow-y-auto">
                        <FeatureVisualization
                            model={selectedModel}
                            feature={selectedFeature}
                            question={selectedQuestion}
                        />
                    </div>
                </div>
            </div>
        </div>
    )
}
