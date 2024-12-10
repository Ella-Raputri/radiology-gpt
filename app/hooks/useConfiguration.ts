"use client"

import { useState, useEffect } from 'react';

const useConfiguration = () => {
  // Safely get values from localStorage
  const getLocalStorageValue = (key: string, defaultValue: any) => {
    if (typeof window !== 'undefined') {
      const storedValue = localStorage.getItem(key);
      if (storedValue !== null) {
        return storedValue;
      }
    }
    return defaultValue;
  };

  const [useRag, setUseRag] = useState<boolean>(() => getLocalStorageValue('useRag', 'true') === 'true');
  const [llm, setLlm] = useState<string>(() => getLocalStorageValue('llm', 'gpt-4o-mini'));

  const setConfiguration = (rag: boolean, llm: string) => {
    setUseRag(rag);
    setLlm(llm);
  }

  // Persist to localStorage
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('useRag', JSON.stringify(useRag));
      localStorage.setItem('llm', llm);
    }
  }, [useRag, llm]);

  return {
    useRag,
    llm,
    setConfiguration,
  };
}

export default useConfiguration;
