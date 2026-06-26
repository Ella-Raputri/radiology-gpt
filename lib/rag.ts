import { Pinecone } from "@pinecone-database/pinecone";

export interface RadiologyFields {
  source: string;
  page: number;
  text: string;
  category: string;
  creationdate?: string;
  creator?: string;
  moddate?: string;
  page_label?: string;
  producer?: string;
  total_pages?: number;
}

const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY
});

const index = pinecone.index({ 
    name: process.env.PINECONE_INDEX_NAME!
});

export async function retrieve(query: string) {
    const result = await index.searchRecords({
        query: {
        topK: 5,
        inputs: {
            text: query,
        },
        },
    });

    return result.result.hits;
}