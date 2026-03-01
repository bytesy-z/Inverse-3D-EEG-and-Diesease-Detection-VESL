"use client"

import type React from "react"
import { useState, useRef, useCallback } from "react"
import { Upload, FileText, X } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

interface FileUploadSectionProps {
  /** Called when a valid file is selected (after client-side validation) */
  onFileSelect: (file: File) => void
  /** Accepted file extensions, e.g. [".edf", ".mat"] */
  accept?: string[]
  /** Short hint shown below the drop zone */
  hint?: string
  /** Disable interaction while processing */
  disabled?: boolean
}

/**
 * Drag-and-drop file upload card with validation feedback.
 *
 * - Shows a dashed drop-zone that highlights on dragover
 * - Validates extension client-side before calling `onFileSelect`
 * - Displays the selected file name + size with a clear button
 * - Fully keyboard-accessible (tab to browse button)
 */
export function FileUploadSection({
  onFileSelect,
  accept = [".edf", ".mat"],
  hint,
  disabled = false,
}: FileUploadSectionProps) {
  const [isDragActive, setIsDragActive] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [validationError, setValidationError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const validate = useCallback(
    (file: File): boolean => {
      const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase()
      if (!accept.includes(ext)) {
        setValidationError(
          `Unsupported format "${ext}". Accepted: ${accept.join(", ")}`
        )
        return false
      }
      setValidationError(null)
      return true
    },
    [accept]
  )

  const handleFile = useCallback(
    (file: File) => {
      if (validate(file)) {
        setSelectedFile(file)
        onFileSelect(file)
      }
    },
    [validate, onFileSelect]
  )

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (disabled) return
    setIsDragActive(e.type === "dragenter" || e.type === "dragover")
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)
    if (disabled) return
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
    // Reset so re-selecting same file triggers onChange
    e.target.value = ""
  }

  const clearFile = () => {
    setSelectedFile(null)
    setValidationError(null)
  }

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  return (
    <Card
      className={`
        relative overflow-hidden border-2 border-dashed transition-colors
        ${disabled ? "pointer-events-none opacity-50" : "cursor-pointer"}
        ${isDragActive
          ? "border-primary bg-primary/5"
          : validationError
            ? "border-destructive/50 bg-destructive/5"
            : "border-border hover:border-primary/50 hover:bg-muted/30"
        }
      `}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={() => !selectedFile && inputRef.current?.click()}
      role="button"
      tabIndex={0}
      aria-label="Upload EEG file"
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault()
          inputRef.current?.click()
        }
      }}
    >
      <div className="flex flex-col items-center justify-center px-6 py-12 text-center">
        {selectedFile ? (
          /* ---- Selected file display ---- */
          <div className="flex items-center gap-3">
            <FileText className="h-5 w-5 text-primary" />
            <div className="text-left">
              <p className="text-sm font-medium text-foreground">
                {selectedFile.name}
              </p>
              <p className="text-xs text-muted-foreground">
                {formatSize(selectedFile.size)}
              </p>
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="ml-2 h-7 w-7 p-0 text-muted-foreground hover:text-destructive"
              onClick={(e) => {
                e.stopPropagation()
                clearFile()
              }}
              aria-label="Remove file"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        ) : (
          /* ---- Empty drop-zone ---- */
          <>
            <div className="mb-3 rounded-lg bg-muted/50 p-3">
              <Upload className="h-6 w-6 text-muted-foreground" />
            </div>
            <p className="text-sm font-medium text-foreground">
              Drop your EEG file here, or{" "}
              <span className="text-primary underline underline-offset-2">
                browse
              </span>
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {hint || `Accepted formats: ${accept.join(", ")}`}
            </p>
          </>
        )}

        {/* Validation error */}
        {validationError && (
          <p className="mt-3 text-xs text-destructive" role="alert">
            {validationError}
          </p>
        )}
      </div>

      <input
        ref={inputRef}
        type="file"
        accept={accept.join(",")}
        onChange={handleInputChange}
        className="hidden"
        aria-hidden="true"
      />
    </Card>
  )
}
