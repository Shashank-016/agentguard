interface AlertBadgeProps {
  count: number;
}

export function AlertBadge({ count }: AlertBadgeProps) {
  if (count === 0) return null;
  return (
    <span className="ml-2 inline-flex items-center justify-center w-5 h-5 text-xs font-bold bg-red-600 text-white rounded-full">
      {count > 99 ? "99+" : count}
    </span>
  );
}
