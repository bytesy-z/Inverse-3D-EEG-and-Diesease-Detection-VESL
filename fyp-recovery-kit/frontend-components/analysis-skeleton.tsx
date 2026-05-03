export function AnalysisSkeleton() {
  return (
    <div className="space-y-6 p-6 animate-pulse">
      <div className="h-8 bg-muted rounded w-1/3" />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="h-[400px] bg-muted rounded" />
        <div className="space-y-4">
          <div className="h-6 bg-muted rounded w-1/2" />
          <div className="h-4 bg-muted rounded w-3/4" />
          <div className="h-4 bg-muted rounded w-2/3" />
          <div className="h-[200px] bg-muted rounded mt-4" />
        </div>
      </div>
    </div>
  )
}
