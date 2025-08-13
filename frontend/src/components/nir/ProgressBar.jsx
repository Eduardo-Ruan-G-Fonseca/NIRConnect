export default function ProgressBar({ percent = 0 }) {
  return (
    <div className="w-full bg-gray-200 rounded">
      <div
        className="h-2 bg-green-500 transition-all"
        style={{ width: `${percent}%` }}
      />
    </div>
  );
}
