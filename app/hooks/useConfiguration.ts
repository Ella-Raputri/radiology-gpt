"use client";

import { useState, useEffect } from "react";

const useConfiguration = () => {
  const [useRag, setUseRag] = useState(true);
  const [llm, setLlm] = useState("gpt-4o-mini");
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const storedRag = localStorage.getItem("useRag");
    const storedLlm = localStorage.getItem("llm");

    if (storedRag !== null) {
      setUseRag(storedRag === "true");
    }

    if (storedLlm !== null) {
      setLlm(storedLlm);
    }
    setLoaded(true);
  }, []);

  useEffect(() => {
    localStorage.setItem("useRag", String(useRag));
    localStorage.setItem("llm", llm);
  }, [useRag, llm]);

  const setConfiguration = (rag: boolean, model: string) => {
    setUseRag(rag);
    setLlm(model);
  };

  return {
    useRag,
    llm,
    loaded,
    setConfiguration,
  };
};

export default useConfiguration;