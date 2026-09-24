import { useEffect, useRef, useState } from 'react'

export function useScrollProgress<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const element = ref.current
    if (!element) return
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
    let frame = 0
    const update = () => {
      frame = 0
      if (reducedMotion.matches) {
        setProgress(1)
        return
      }
      const bounds = element.getBoundingClientRect()
      const distance = window.innerHeight * 0.8 + bounds.height * 0.45
      const next = Math.max(0, Math.min(1, (window.innerHeight * 0.9 - bounds.top) / distance))
      setProgress((previous) => Math.abs(previous - next) > 0.005 ? next : previous)
    }
    const schedule = () => { if (!frame) frame = window.requestAnimationFrame(update) }
    update()
    window.addEventListener('scroll', schedule, { passive: true })
    window.addEventListener('resize', schedule)
    reducedMotion.addEventListener('change', schedule)
    return () => {
      window.cancelAnimationFrame(frame)
      window.removeEventListener('scroll', schedule)
      window.removeEventListener('resize', schedule)
      reducedMotion.removeEventListener('change', schedule)
    }
  }, [])

  return { ref, progress }
}
