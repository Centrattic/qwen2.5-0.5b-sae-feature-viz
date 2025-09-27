import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
    title: 'SAE Feature Visualization',
    description: 'Explore SAE features in Qwen2.5-0.5B models',
}

export default function RootLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <html lang="en">
            <body className="bg-gray-50 min-h-screen">
                {children}
            </body>
        </html>
    )
}
