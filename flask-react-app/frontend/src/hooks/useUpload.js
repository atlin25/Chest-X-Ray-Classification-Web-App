import { useState } from 'react';

export function useUpload() {
  const [loading, setLoading] = useState(false);

  const upload = async ({ file }) => {
    setLoading(true);

    // fake upload logic
    const url = URL.createObjectURL(file);

    setLoading(false);
    return { url, error: null };
  };

  return [upload, { loading }];
}

