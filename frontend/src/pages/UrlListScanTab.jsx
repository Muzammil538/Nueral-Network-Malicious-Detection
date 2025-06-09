import React, { useState } from "react";
import axios from "axios";
import { Bar } from "react-chartjs-2";
import { Chart, BarElement, CategoryScale, LinearScale } from "chart.js";

Chart.register(BarElement, CategoryScale, LinearScale);

function UrlListScanTab() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState([]);
  const [barGraph, setBarGraph] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResults([]);
    setBarGraph(null);
    setError("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setResults([]);
    setBarGraph(null);
    setError("");
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await axios.post("http://localhost:5000/scan/urllist", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResults(res.data.results);
      setBarGraph(res.data.bar_graph);
    } catch (err) {
      setError("Scan failed. Please try again.");
    }
    setLoading(false);
  };

  return (
    <div className="pt-10 min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-blue-900 flex flex-col">
      <main className="flex flex-1 flex-col items-center justify-center px-4">
        <div className="bg-gray-800 bg-opacity-90 rounded-xl shadow-lg p-8 w-full max-w-xl flex flex-col items-center">
          <h2 className="text-3xl font-bold text-white mb-4">
            Scan a List of URLs (Excel)
          </h2>
          <form onSubmit={handleSubmit} className="w-full flex flex-col items-center">
            <input
              type="file"
              accept=".xlsx,.xls,.csv"
              onChange={handleFileChange}
              disabled={loading}
              className="mb-4 text-white"
              required
            />
            <button
              type="submit"
              className="w-full py-3 rounded-full bg-gradient-to-r from-blue-500 to-blue-700 text-white font-bold text-lg shadow hover:from-blue-600 hover:to-blue-800 transition mb-2"
              disabled={loading || !file}
            >
              {loading ? "Scanning..." : "Scan URL List"}
            </button>
          </form>
        </div>
        {error && (
          <div className="mt-4 text-red-400 bg-gray-900 rounded p-4">{error}</div>
        )}
        {barGraph && (
          <div className="mt-8 w-full max-w-md bg-gray-800 bg-opacity-90 rounded-xl shadow-lg p-6">
            <h3 className="text-white text-lg mb-4">Safe vs Unsafe URLs</h3>
            <Bar
              data={{
                labels: ["Safe", "Unsafe"],
                datasets: [
                  {
                    label: "Count",
                    data: [barGraph.safe, barGraph.unsafe],
                    backgroundColor: ["#22c55e", "#ef4444"],
                  },
                ],
              }}
              options={{
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                  y: { beginAtZero: true, ticks: { color: "#fff" } },
                  x: { ticks: { color: "#fff" } },
                },
              }}
            />
          </div>
        )}
        {results.length > 0 && (
          <div className="mt-8 w-full max-w-2xl bg-gray-800 bg-opacity-90 rounded-xl shadow-lg p-6">
            <h3 className="text-white text-lg mb-4">Scan Results</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full text-white">
                <thead>
                  <tr>
                    <th className="px-4 py-2">URL</th>
                    <th className="px-4 py-2">Prediction</th>
                    <th className="px-4 py-2">Confidence</th>
                    <th className="px-4 py-2">Image</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r, idx) => (
                    <tr key={idx}>
                      <td className="px-4 py-2 break-all">{r.url}</td>
                      <td className={`px-4 py-2 font-bold ${r.prediction === "safe" ? "text-green-400" : "text-red-400"}`}>
                        {r.prediction.toUpperCase()}
                      </td>
                      <td className="px-4 py-2">{r.confidence}</td>
                      <td className="px-4 py-2">
                        {r.image_url && (
                          <img src={r.image_url} alt="URL" className="w-16 h-16 rounded shadow" />
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
      <footer className="w-full py-4 text-center text-blue-200 text-sm bg-transparent mt-auto">
        &copy; {new Date().getFullYear()} Malicious Detection. All rights reserved.
      </footer>
    </div>
  );
}

export default UrlListScanTab;