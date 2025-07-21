'use client';

import { useState } from 'react';

export default function Home() {
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);

    const formData = new FormData(e.currentTarget);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
      setResult({ success: false, error: 'Failed to upload file' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="file" className="block mb-2">Select File:</label>
          <input
            type="file"
            name="file"
            id="file"
            required
            className="border p-2"
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:opacity-50"
        >
          {loading ? 'Processing...' : 'Upload File'}
        </button>
      </form>

      {result && (
        <div className="mt-8">
          {result.image && (
            <div>
              <h3 className="text-lg mb-2">Generated Image:</h3>
              <img
                src={`data:image/png;base64,${result.image}`}
                alt="Analysis Result"
                className="max-w-full h-auto"
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
