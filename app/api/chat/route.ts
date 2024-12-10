import OpenAI from 'openai';
import { OpenAIStream, StreamingTextResponse } from 'ai';
import { AstraDB } from "@datastax/astra-db-ts";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const astraDb = new AstraDB(
  process.env.ASTRA_DB_APPLICATION_TOKEN,
  process.env.ASTRA_DB_API_ENDPOINT,
  process.env.ASTRA_DB_NAMESPACE
);

export async function POST(req: Request) {
  try {
    const {messages, useRag, llm} = await req.json();

    const latestMessage = messages[messages?.length - 1]?.content;

    let docContext = '';
    if (useRag) {
      // Generate embedding for the user's query
      const {data} = await openai.embeddings.create({
        input: latestMessage,
        model: 'text-embedding-3-small'
      });

      // Use the single collection "chat_embeddings"
      const collection = await astraDb.collection("chat_radiology");

      // Perform a vector similarity search
      const cursor = collection.find(null, {
        sort: {
          $vector: data[0]?.embedding,
        },
        limit: 5,
      });

      const documents = await cursor.toArray();
      
      // Join the top documents as context
      docContext = documents?.map(doc => doc.text).join("\n") || '';
    }
    
    const ragPrompt = [
      {
        role: 'system',
        content: `Anda adalah asisten medis yang membantu menjawab pertanyaan berdasarkan informasi berikut:
        START CONTEXT
        ${docContext}
        END CONTEXT

        Jawablah pertanyaan berikut berdasarkan informasi yang diberikan kepada Anda.
        Apabila informasi berupa gejala-gejala yang dialami, berikan lima kemungkinan diagnosa berdasarkan informasi tersebut, dan untuk setiap kemungkinan diagnosis, berikan penjelasan singkat yang menghubungkan gejala dengan diagnosis tersebut.
        Berikan saran kepada pasien hal apa saja yang harus dilakukan selanjutnya.
        Apabila informasi bukan berupa gejala-gejala, jawablah sesuai konteks pertanyaan tersebut dan usahakan memberikan sumber dengan menuliskan nama file dari mana Anda mendapatkan jawabannya.
        Jika Anda tidak yakin dengan jawabannya dan/atau pertanyaan belum terlalu spesifik, katakan bahwa Anda tidak yakin dengan jawaban Anda.
        Jangan menambahkan informasi di luar konteks yang diberikan.
        Pastikan jawaban Anda akurat dan berdasarkan pedoman klinis resmi.
        Berikan peringatan bahwa ini bukan pengganti nasihat medis profesional dan pasien harus berkonsultasi dengan dokter untuk diagnosis yang akurat`,
      },
    ];

    const response = await openai.chat.completions.create(
      {
        model: llm ?? 'gpt-4o-mini',
        stream: true,
        messages: [...ragPrompt, ...messages],
      }
    );

    const stream = OpenAIStream(response);
    return new StreamingTextResponse(stream);
  } catch (e) {
    throw e;
  }
}
