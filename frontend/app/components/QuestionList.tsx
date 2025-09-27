'use client'

interface QuestionListProps {
    questions: string[]
    selectedQuestion: string
    onQuestionSelect: (question: string) => void
}

export default function QuestionList({ questions, selectedQuestion, onQuestionSelect }: QuestionListProps) {
    return (
        <div className="space-y-2 max-h-64 overflow-y-auto">
            {questions.map((question, index) => (
                <div
                    key={index}
                    className={`p-3 rounded-lg cursor-pointer transition-colors ${selectedQuestion === question
                            ? 'bg-primary-100 border border-primary-300'
                            : 'bg-gray-50 hover:bg-gray-100 border border-gray-200'
                        }`}
                    onClick={() => onQuestionSelect(question)}
                >
                    <p className="text-sm text-gray-700 line-clamp-3">
                        {question}
                    </p>
                </div>
            ))}
        </div>
    )
}
