import React, { useState } from "react";
import axios from "axios";
import { Link } from "react-router";

function UrlScanTab() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResult(null);
    setLoading(true);
    try {
      const res = await axios.post("http://localhost:5000/scan/url", { url });
      setResult(res.data);
    } catch (err) {
      setResult({ error: "Scan failed. Please try again. " });
      console.log("Error is : ", err);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-blue-900 flex flex-col">
      {/* Main Content */}
      <main className="flex flex-1 flex-col items-center justify-center px-4">
        <div className="bg-gray-800 bg-opacity-90 rounded-xl shadow-lg p-8 w-full max-w-xl flex flex-col items-center">
          <h2 className="text-3xl font-bold text-white mb-4">
            Scan a URL for Malicious Threats
          </h2>
          <form
            onSubmit={handleSubmit}
            className="w-full flex flex-col items-center"
          >
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              disabled={loading}
              placeholder="Enter URL to scan"
              className="w-full px-4 py-3 rounded-lg mb-4 bg-gray-900 text-white border border-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-400"
              required
            />
            <button
              type="submit"
              className="w-full py-3 rounded-full bg-gradient-to-r from-blue-500 to-blue-700 text-white font-bold text-lg shadow hover:from-blue-600 hover:to-blue-800 transition mb-2"
              disabled={loading}
            >
              {loading ? "Scanning..." : "Scan URL"}
            </button>
          </form>
        </div>
        {result && !result.error && (
          <div className="mt-4 text-center bg-gray-800 bg-opacity-90 rounded-xl shadow-lg p-8 w-full max-w-xl flex flex-col items-center">
            <p className="text-lg text-white">
              Prediction:{" "}
              <span
                className={
                  result.prediction === "benign"
                    ? "text-green-500"
                    : "text-red-500 font-semibold"
                }
              >
                {result.prediction.toUpperCase()}
              </span>{" "}
            </p>

            {result.image_url && (
              <div className="mt-4">
                <h3 className="text-white mb-2 text-lg">
                  TF-IDF Image Representation:
                </h3>
                <img
                  src={result.image_url}
                  alt="URL grayscale image"
                  className="w-64 h-64 border rounded shadow"
                />
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="w-full py-4 text-center text-blue-200 text-sm bg-transparent mt-auto">
        &copy; {new Date().getFullYear()} Malicious Detection. All rights
        reserved.
      </footer>
    </div>
  );
}

export default UrlScanTab;
