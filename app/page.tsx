"use client";
import {useEffect, useRef, useState} from 'react';
import Bubble from '../components/Bubble'
import { useChat, Message } from 'ai/react';
import Footer from '../components/Footer';
import Configure from '../components/Configure';
import PromptSuggestionRow from '../components/PromptSuggestions/PromptSuggestionsRow';
import ThemeButton from '../components/ThemeButton';
import useConfiguration from './hooks/useConfiguration';

export default function Home() {
  const { append, messages, input, handleInputChange, handleSubmit } = useChat();
  const { useRag, llm, setConfiguration } = useConfiguration();
  
  const messagesEndRef = useRef(null);
  const [configureOpen, setConfigureOpen] = useState(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = (e) => {
    handleSubmit(e, { options: { body: { useRag, llm } } });
  }

  const handlePrompt = (promptText) => {
    const msg: Message = { id: crypto.randomUUID(), content: promptText, role: 'user' };
    append(msg, { options: { body: { useRag, llm } } });
  };

  return (
    <div>Under maintenance</div>
  )
}
