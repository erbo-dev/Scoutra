type BrandMarkProps = {
  compact?: boolean
  className?: string
}

export function BrandMark({ compact = false, className = '' }: BrandMarkProps) {
  return <span className={`brand-mark ${compact ? 'brand-mark--compact' : ''} ${className}`}>SCOUTRA</span>
}
