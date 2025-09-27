'use client'

interface QuestionListProps {
    questions: string[]
    selectedQuestion: string
    onQuestionSelect: (question: string) => void
}

export default function QuestionList({ questions, selectedQuestion, onQuestionSelect }: QuestionListProps) {
    return (
        <div className="space-y-1 max-h-48 overflow-y-auto">
            {questions.map((question, index) => (
                <div
                    key={index}
                    className={`p-2 rounded cursor-pointer transition-colors text-xs ${selectedQuestion === question
                        ? 'bg-blue-100 border border-blue-300 text-blue-900'
                        : 'bg-gray-50 hover:bg-gray-100 border border-gray-200 text-gray-700'
                        }`}
                    onClick={() => onQuestionSelect(question)}
                >
                    <p className="line-clamp-2 leading-tight">
                        {question}
                    </p>
                </div>
            ))}
        </div>
    )
}
