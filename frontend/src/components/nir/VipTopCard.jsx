// frontend/src/components/nir/VipTopCard.jsx
import React from "react";

export default function VipTopCard({ vip = [], top = 30, title = "VIPs (Top)" }) {
  const topVip = vip.slice(0, top);

  if (!topVip.length) {
    return (
      <div className="card dashed h-64 flex items-center justify-center">
        <p>Sem VIPs disponíveis para esta calibração.</p>
      </div>
    );
  }

  return (
    <div className="card p-4">
      <h3 className="card-title mb-3">{title}</h3>
      <div className="overflow-x-auto">
        <table className="table table-sm">
          <thead>
            <tr>
              <th>#</th>
              <th>λ (nm)</th>
              <th>VIP</th>
            </tr>
          </thead>
          <tbody>
            {topVip.map((d, i) => (
              <tr key={i}>
                <td>{i + 1}</td>
                <td>{Number(d.wavelength).toFixed(2)}</td>
                <td>{Number(d.score).toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
