import OpenAI from 'openai';
import { OpenAIStream, StreamingTextResponse } from 'ai';
import { AstraDB } from "@datastax/astra-db-ts";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const astraDb = new AstraDB(
  process.env.ASTRA_DB_APPLICATION_TOKEN!,
  process.env.ASTRA_DB_API_ENDPOINT!,
  process.env.ASTRA_DB_NAMESPACE!
);

export async function POST(req: Request) {
  try {
    const { messages, useRag, llm } = await req.json();

    const latestMessage = messages[messages?.length - 1]?.content;
    let docContext = '';

    if (useRag) {
      // Generate embedding for the user's query
      const { data } = await openai.embeddings.create({
        input: latestMessage,
        model: 'text-embedding-3-small'
      });

      // Use the single collection "chat_radiology"
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
      // If documents contain metadata like filename, ensure each doc has a reference to it.
      // For now, we assume 'document_id' field exists or we add disclaimers if not.
      docContext = documents?.map(doc => {
        const filename = doc.category || "dokumen tidak diketahui";
        return `Sumber: ${filename}\n${doc.text}\n`;
      }).join("\n") || '';
    }

    // Enhanced instructions for thoroughness and detail
    const ragPrompt = [
      {
        role: 'system',
        content: `
        Anda adalah asisten medis yang membantu menjawab pertanyaan berdasarkan informasi berikut:
        START CONTEXT
        ${docContext}
        END CONTEXT

        Instruksi:
        1. Jawablah pertanyaan secara panjang, terstruktur, dan sangat mendetail.
        2. Apabila informasi berupa gejala-gejala yang dialami pasien, berikan tiga kemungkinan diagnosis berdasarkan informasi tersebut, dengan penjelasan singkat yang menghubungkan gejala dengan setiap diagnosis. Kemudian, berikan saran kepada dokter untuk investigasi gejala lebih lanjut berdasarkan informasi dalam dokumen, digabungkan dengan pengetahuan Anda.
        3. Jika pertanyaan berkaitan dengan suatu penyakit (misalnya "Pasien terkena TB Paru. Jelaskan..."), berikan penjelasan selengkap dan sedetail mungkin tentang penyakit tersebut, mengacu secara langsung pada informasi yang ada dalam konteks (dokumen). Sertakan informasi tentang etiologi, gejala khas, diagnosa, penatalaksanaan, dan referensi dokumen. Jika tersedia, sebutkan nama file asal informasi tersebut.
        4. Apabila informasi bukan berupa gejala tetapi pertanyaan umum, jawablah secara lengkap dan rujuk sumber informasi dari dokumen dengan menyebutkan nama file.
        5. Jika Anda tidak yakin dengan jawaban (informasi tidak terdapat dalam konteks) atau pertanyaan kurang spesifik, katakan Anda tidak yakin dengan jawaban Anda.
        6. Jangan menambahkan informasi di luar konteks yang diberikan.
        7. Pastikan jawaban akurat, berdasarkan pedoman klinis resmi atau dokumen yang tersedia.
        8. Berikan peringatan bahwa ini bukan pengganti nasihat medis profesional dan pasien harus berkonsultasi dengan dokter untuk diagnosis dan perawatan yang akurat.
        9. Di akhir jawaban, sampaikan bahwa jawaban diambil berdasarkan Keputusan Menteri Kesehatan Republik Indonesia Nomor HK.01.07/MENKES/1186/2022 tentang Panduan Praktik Klinis bagi Dokter di Fasilitas Pelayanan Kesehatan Tingkat Pertama.

        Pastikan jawaban mencakup seluruh konteks relevan dari dokumen yang diberikan. Susun jawaban dengan paragraf yang jelas, dan cantumkan nama file dokumen jika tersedia. Berikan detail yang selengkap-lengkapnya.`
      },
    ];

    const response = await openai.chat.completions.create(
      {
        model: llm ?? 'gpt-4o-mini',
        stream: true,
        messages: [...ragPrompt, ...messages],
        max_tokens: 3000, // Allow for a longer answer
        temperature: 0.2  // Lower temperature for more detailed and focused response
      }
    );

    const stream = OpenAIStream(response);
    return new StreamingTextResponse(stream);
  } catch (e) {
    throw e;
  }
}
