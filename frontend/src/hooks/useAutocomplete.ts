import { useState, useEffect, useRef } from 'react'

export function useAutocomplete(prefix: string, enabled: boolean = true) {
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [open,        setOpen]        = useState(false)
  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (!enabled || !prefix || prefix.length < 2) {
      setSuggestions([])
      setOpen(false)
      return
    }

    if (debounce.current) clearTimeout(debounce.current)
    debounce.current = setTimeout(async () => {
      try {
        const res  = await fetch(`/autocomplete?q=${encodeURIComponent(prefix)}&n=8`)
        const data = await res.json() as { suggestions?: string[] }
        setSuggestions(data.suggestions ?? [])
        setOpen((data.suggestions ?? []).length > 0)
      } catch {
        setSuggestions([])
        setOpen(false)
      }
    }, 160)

    return () => { if (debounce.current) clearTimeout(debounce.current) }
  }, [prefix, enabled])

  const close = () => setOpen(false)

  return { suggestions, open, close }
}
