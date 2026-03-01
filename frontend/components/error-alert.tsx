"use client"

import { AlertCircle, RotateCcw } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"

interface ErrorAlertProps {
  message: string
  onRetry: () => void
}

/**
 * Full-width error banner with retry action.
 */
export function ErrorAlert({ message, onRetry }: ErrorAlertProps) {
  return (
    <Alert variant="destructive" className="mx-auto max-w-lg">
      <AlertCircle className="h-4 w-4" />
      <AlertTitle>Processing Error</AlertTitle>
      <AlertDescription className="mt-1 text-sm">{message}</AlertDescription>
      <Button
        variant="outline"
        size="sm"
        className="mt-4"
        onClick={onRetry}
      >
        <RotateCcw className="mr-2 h-3.5 w-3.5" />
        Try Again
      </Button>
    </Alert>
  )
}
