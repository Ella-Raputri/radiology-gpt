import { Anthropic } from "@anthropic-ai/sdk";
import { AstraDB } from "@datastax/astra-db-ts";
import OpenAI from 'openai';
import { OpenAIStream, StreamingTextResponse } from 'ai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
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
        ${useRag ? `START CONTEXT
        ${docContext}
        END CONTEXT` : ''}

        Instruksi:
        1. Jawablah pertanyaan secara panjang, terstruktur, dan sangat mendetail.
        2. Apabila informasi berupa gejala-gejala yang dialami pasien, berikan lima kemungkinan diagnosis berdasarkan informasi tersebut, dengan penjelasan singkat yang menghubungkan gejala dengan setiap diagnosis. Kemudian, berikan saran kepada dokter untuk investigasi gejala lebih lanjut berdasarkan informasi dari dalam dokumen.
        3. Jika gejalanya menyangkut tentang nyeri atau sakit di ulu hati atau perut bagian atas, selalu berikan dispepsia sebagai salah satu diagnosis kemungkinan.
        4. Jika gejala yang disampaikan berhubungan dengan demam, seperti demam tinggi atau demam lebih dari 2 hari, selalu berikan demam biasa sebagai salah satu diagnosis kemungkinan, dan sertakan informasi tentang demam sedikit lebih detail seperti cara obat yang diperlukan.
        5. Jika gejala yang disampaikan berhubungan dengan nafas sesak atau batuk, kemungkinan besar pasien terkena penyakit asthma. Jadi, selalu berikan asthma sebagai salah satu diagnosis kemungkinan.
        6. Jika pertanyaan berkaitan dengan suatu penyakit (misalnya "Pasien terkena TB Paru. Jelaskan..."), berikan penjelasan selengkap dan sedetail mungkin tentang penyakit tersebut, mengacu secara langsung pada informasi yang ada dalam konteks (dokumen). Sertakan informasi tentang etiologi, gejala khas, diagnosa, penatalaksanaan, dan referensi dokumen. Jika tersedia, sebutkan nama file asal informasi tersebut.
        7. Apabila informasi bukan berupa gejala tetapi pertanyaan umum, jawablah secara lengkap dan rujuk sumber informasi dari dokumen dengan menyebutkan nama file.
        8. Jika Anda tidak yakin dengan jawaban (informasi tidak terdapat dalam konteks) atau pertanyaan kurang spesifik, katakan Anda tidak yakin dengan jawaban Anda.
        9. Jangan menambahkan informasi di luar konteks yang diberikan.
        10. Pastikan jawaban akurat, berdasarkan pedoman klinis resmi atau dokumen yang tersedia.
        11. Berikan peringatan bahwa ini bukan pengganti nasihat medis profesional dan pasien harus berkonsultasi dengan dokter untuk diagnosis dan perawatan yang akurat.
        ${ useRag ? '12. Di akhir jawaban, sampaikan bahwa jawaban diambil berdasarkan Keputusan Menteri Kesehatan Republik Indonesia Nomor HK.01.07/MENKES/1186/2022 tentang Panduan Praktik Klinis bagi Dokter di Fasilitas Pelayanan Kesehatan Tingkat Pertama.' : '' }

        Pastikan jawaban mencakup seluruh konteks relevan dari dokumen yang diberikan. Susun jawaban dengan paragraf yang jelas, dan cantumkan nama file dokumen jika tersedia. Berikan detail yang selengkap-lengkapnya.`
      },
    ];

    // Determine which LLM to use
    console.log('Using LLM:', llm);
    if (llm === 'gpt-4o-mini' || !llm) {
      // OpenAI (GPT-4o-mini) implementation
      const response = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        stream: true,
        messages: [...ragPrompt, ...messages],
        max_tokens: 5000,
        temperature: 0.2
      });

      const stream = OpenAIStream(response);
      return new StreamingTextResponse(stream);
    } else if (llm.startsWith('claude-')) {
      const anthropicModel = (() => {
        switch (llm) {
          case 'claude-4-1-opus':
            return 'claude-opus-4-1-20250805';
          case 'claude-4-sonnet':
            return 'claude-sonnet-4-20250514';
          case 'claude-3-5-haiku':
          default:
            return 'claude-3-5-haiku-20241022';
        }
      })();

      console.log('Using Anthropic model:', anthropicModel);

      const stream = await anthropic.messages.create({
        model: anthropicModel,
        max_tokens: 4096,
        messages: [
          { role: 'user', content: ragPrompt[0].content + '\n\n' + messages[messages.length - 1].content }
        ],
        stream: true
      });
      console.log('Anthropic stream:', stream);

      const textStream = new ReadableStream({
        async start(controller) {
          for await (const messageStream of stream) {
            if (messageStream.type === 'content_block_delta' && 'delta' in messageStream) {
              const delta = messageStream.delta;
              if ('text' in delta) {
                controller.enqueue(delta.text);
              }
            }
          }
          controller.close();
        }
      });

      console.log('Anthropic text stream:', textStream);

      return new StreamingTextResponse(textStream);
    }
    else if (llm === 'gpt-4o' || !llm) {
      // OpenAI (GPT-4o-mini) implementation
      const response = await openai.chat.completions.create({
        model: 'gpt-4o',
        stream: true,
        messages: [...ragPrompt, ...messages],
        max_tokens: 5000,
        temperature: 0.2
      });

      const stream = OpenAIStream(response);
      return new StreamingTextResponse(stream); 
    }
    else if (llm === 'deepseek-v3' || llm === 'qwen-2-5-72b' || !llm) {
      console.log(`Using ${llm} via OpenRouter`);
      const llmModel = (() => {
        switch (llm) {
          case 'qwen-2-5-72b':
            return "qwen/qwen-2.5-72b-instruct:free"; //qwen
          default:
            return "deepseek/deepseek-chat-v3.1:free"; //deepseek V3
        }
      })();

      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          "model": llmModel,
          "messages": [
            {
              "role": "user",
              "content": ragPrompt[0].content + '\n\n' + messages[messages.length - 1].content
            }
          ],
          "stream": true,
          "max_tokens": 5000,
          "temperature": 0.2
        })
      });

      if (!response.ok) {
        console.error('DeepSeek API error:', await response.text());
        throw new Error('Failed to fetch response from DeepSeek');
      }

      const stream = new ReadableStream({
        async start(controller) {
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
    
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              controller.close();
              break;
            }
    
            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n').filter(line => line.trim() !== '');
            
            let buffer = '';
            for (const line of lines) {
              if (line.startsWith(':')) continue;

              if (line.startsWith('data:')) {
                const message = line.replace(/^data: /, '').trim();
                if (message === '[DONE]') continue;
                buffer += message;

                try {
                  const parsed = JSON.parse(buffer);
                  buffer = '';

                  if (parsed.choices?.[0]?.delta?.content) {
                    controller.enqueue(parsed.choices[0].delta.content);
                  }
                } catch (error) {
                  console.error('Error parsing JSON:', error, 'Message:', message);
                }
              }
            }
          }
        }
      });
    
      return new StreamingTextResponse(stream);
    }
    else {
      throw new Error(`Unsupported LLM: ${llm}`);
    }
  } catch (e) {
    console.error('Error in medical AI processing:', e);
    throw e;
  }
}
