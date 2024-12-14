import { useState } from "react";
import Dropdown from "./Dropdown";
import Toggle from "./Toggle";
import Footer from "./Footer";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  useRag: boolean;
  llm: string;
  setConfiguration: (useRag: boolean, llm: string) => void;
}

const Configure = ({ isOpen, onClose, useRag, llm, setConfiguration }: Props) => {
  const [rag, setRag] = useState(useRag);
  const [selectedLlm, setSelectedLlm] = useState(llm);

  if (!isOpen) return null;

  const llmOptions = [
    { label: 'GPT-4o mini', value: 'gpt-4o-mini' },
    { label: 'GPT-4o', value: 'gpt-4o' },
    {label: 'Claude 3.5 Haiku', value: 'claude-3-5-haiku'},
    {label: 'Claude 3 opus', value: 'claude-3-opus'},
    {label: 'Claude 3.5 sonnet', value: 'claude-3-5-sonnet'}

  ];

  const handleSave = () => {
    setConfiguration(rag, selectedLlm);
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-75 flex items-center justify-center z-50">
      <div className="chatbot-section flex flex-col origin:w-[800px] w-full origin:h-[735px] h-full p-6 rounded shadow-lg overflow-auto">
        <div className="grow">
          <div className='pb-6 flex justify-between'>
            <h1 className='chatbot-text-primary text-xl md:text-2xl font-medium'>Configure</h1>
            <button
              onClick={onClose}
              className="chatbot-text-primary text-4xl font-thin leading-8"
            >
              <span aria-hidden>x</span>
            </button>
          </div>
          <div className="flex mb-4 gap-4">
            <Dropdown
              fieldId="llm"
              label="LLM"
              options={llmOptions}
              value={selectedLlm}
              onSelect={setSelectedLlm}
            />
            <Toggle enabled={rag} label="Enable vector content (RAG)" onChange={() => setRag(!rag)} />
          </div>
        </div>
        <div className="self-end w-full">
          <div className="flex justify-end gap-2">
            <button
              className='chatbot-button-secondary flex rounded-md items-center justify-center px-2.5 py-3'
              onClick={onClose}
            >
              Cancel
            </button>
            <button
              className='chatbot-button-primary flex rounded-md items-center justify-center px-2.5 py-3'
              onClick={handleSave}
            >
              Save Configuration
            </button>
          </div>
          <Footer />
        </div>
      </div>
    </div>
  );
};

export default Configure;
