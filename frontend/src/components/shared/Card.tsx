import clsx from 'clsx'
import type { ReactNode } from 'react'

interface CardProps {
  title?: string
  children: ReactNode
  className?: string
}

export function Card({ title, children, className }: CardProps) {
  return (
    <div className={clsx('card', className)}>
      {title && <h3 className="section-title">{title}</h3>}
      {children}
    </div>
  )
}
