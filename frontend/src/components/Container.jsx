export default function Container({ children, className = "" }) {
  return (
    <div className={`mx-auto w-full max-w-screen-xl px-4 sm:px-6 lg:px-8 ${className}`}>
      {children}
    </div>
  );
}
