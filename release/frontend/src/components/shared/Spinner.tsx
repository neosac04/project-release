import clsx from 'clsx'

export function Spinner({ className }: { className?: string }) {
  return (
    <div
      className={clsx(
        'inline-block rounded-full border-2 border-gray-700 border-t-brand-500 animate-spin',
        className,
      )}
    />
  )
}
