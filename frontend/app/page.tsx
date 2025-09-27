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
            <div className="min-h-screen flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600 mx-auto"></div>
                    <p className="mt-4 text-gray-600">Loading SAE Feature Visualization...</p>
                </div>
            </div>
        )
    }

    return (
        <div className="min-h-screen bg-gray-50">
            <header className="bg-white shadow-sm border-b">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between items-center py-6">
                        <div>
                            <h1 className="text-3xl font-bold text-gray-900">SAE Feature Visualization</h1>
                            <p className="mt-2 text-gray-600">Explore SAE features in Qwen2.5-0.5B models</p>
                        </div>
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Control Panel */}
                    <div className="lg:col-span-1 space-y-6">
                        <div className="bg-white rounded-lg shadow p-6">
                            <h2 className="text-lg font-semibold text-gray-900 mb-4">Model Selection</h2>
                            <ModelSelector
                                models={models}
                                selectedModel={selectedModel}
                                onModelChange={setSelectedModel}
                            />
                        </div>

                        <div className="bg-white rounded-lg shadow p-6">
                            <h2 className="text-lg font-semibold text-gray-900 mb-4">Feature Selection</h2>
                            <FeatureSelector
                                selectedFeature={selectedFeature}
                                onFeatureChange={setSelectedFeature}
                            />
                        </div>

                        <div className="bg-white rounded-lg shadow p-6">
                            <h2 className="text-lg font-semibold text-gray-900 mb-4">Questions</h2>
                            <QuestionList
                                questions={questions}
                                selectedQuestion={selectedQuestion}
                                onQuestionSelect={setSelectedQuestion}
                            />
                        </div>

                        <div className="bg-white rounded-lg shadow p-6">
                            <h2 className="text-lg font-semibold text-gray-900 mb-4">Add Question</h2>
                            <QuestionInput
                                onQuestionSubmit={handleQuestionSubmit}
                                selectedModel={selectedModel}
                            />
                        </div>
                    </div>

                    {/* Visualization */}
                    <div className="lg:col-span-2">
                        <div className="bg-white rounded-lg shadow">
                            <div className="p-6">
                                <h2 className="text-lg font-semibold text-gray-900 mb-4">
                                    Feature Visualization
                                </h2>
                                <FeatureVisualization
                                    model={selectedModel}
                                    feature={selectedFeature}
                                    question={selectedQuestion}
                                />
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    )
}
