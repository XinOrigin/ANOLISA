import React from 'react';

export interface DateTimePickerProps {
  value: number; // ms timestamp
  onChange: (ms: number) => void;
  label?: string;
  className?: string;
}

function toDatetimeLocal(ms: number): string {
  if (!Number.isFinite(ms)) return '';
  const d = new Date(ms);
  if (Number.isNaN(d.getTime())) return '';
  const year = d.getFullYear();
  if (year < 1) return '';
  const pad = (n: number, len = 2) => String(n).padStart(len, '0');
  return `${pad(year, 4)}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function fromDatetimeLocal(val: string): number {
  if (!val) return Number.NaN;
  const localMatch = /^(\d{4,})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,3}))?)?$/.exec(val);
  if (!localMatch) return Number.NaN;
  const year = Number(localMatch[1]);
  const month = Number(localMatch[2]);
  const day = Number(localMatch[3]);
  const hour = Number(localMatch[4]);
  const minute = Number(localMatch[5]);
  const second = Number(localMatch[6] || '0');
  const millisecond = Number((localMatch[7] || '0').padEnd(3, '0'));
  if (year < 1 || month < 1 || month > 12 || day < 1 || day > 31 || hour > 23 || minute > 59 || second > 59) {
    return Number.NaN;
  }
  const d = new Date(0);
  d.setFullYear(year, month - 1, day);
  d.setHours(hour, minute, second, millisecond);
  if (d.getFullYear() !== year || d.getMonth() + 1 !== month || d.getDate() !== day) {
    return Number.NaN;
  }
  return d.getTime();
}

export const DateTimePicker: React.FC<DateTimePickerProps> = ({ value, onChange, label, className }) => {
  return (
    <div className={`flex items-center gap-2 ${className || ''}`}>
      {label && <label className="text-sm text-gray-600 whitespace-nowrap">{label}</label>}
      <input
        type="datetime-local"
        className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
        value={toDatetimeLocal(value)}
        onChange={(e) => {
          if (!e.target.value) return;
          const parsed = fromDatetimeLocal(e.target.value);
          if (Number.isFinite(parsed)) onChange(parsed);
        }}
      />
    </div>
  );
};
