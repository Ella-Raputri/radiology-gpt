import { AstraDB } from "@datastax/astra-db-ts";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import 'dotenv/config';
import OpenAI from 'openai';
import pdf from 'pdf-parse'; 
import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
});

const { ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_API_ENDPOINT, ASTRA_DB_NAMESPACE } = process.env;

const astraDb = new AstraDB(ASTRA_DB_APPLICATION_TOKEN!, ASTRA_DB_API_ENDPOINT!, ASTRA_DB_NAMESPACE!);

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const SIMILARITY_METRIC = 'cosine';
const COLLECTION_NAME = 'chat_embeddings';

const pdf_category_mapping: { [filename: string]: string } = {
    "0_PPK_UMUM.pdf": "Penyakit Umum",
    "1_DARAH_PEMBENTUKAN_DARAH_DAN_SISTEM_IMUN.pdf": "Darah, Pembentukan Darah, dan Sistem Imun",
    "2_DIGESTIVE.pdf": "Digestive",
    "3_MATA.pdf": "Mata",
    "4_TELINGA.pdf": "Digestive",
    "5_KARDIOVASKULER.pdf": "Kardiovaskuler",
    "6_MUSKULOSKELETAL.pdf": "Muskuloskeletal",
    "7_NEUROLOGI.pdf": "Neurologi",
    "8_PSIKIATRI.pdf": "Psikiatri",
    "9_RESPIRASI.pdf": "Respirasi",
    "10_KULIT.pdf": "Kulit",
    "11_METABOLIK_ENDOKRIN_DAN_NUTRISI.pdf": "Metabolik Endokrin dan Nutrisi",
    "12_GINJAL_DAN_SALURAN_KEMIH.pdf": "Ginjal dan Saluran Kemih",
    "13_KESEHATAN_WANITA.pdf": "Kesehatan Wanita",
    "14_PENYAKIT_KELAMIN.pdf": "Penyakit Kelamin"
};

async function createCollection() {
  try {
    const res = await astraDb.createCollection(COLLECTION_NAME, {
      vector: {
        dimension: 1536,
        metric: SIMILARITY_METRIC
      }
    });
    console.log(res);
  } catch (e) {
    console.log(`${COLLECTION_NAME} already exists`);
  }
}

async function loadPDFsFromDirectory(dir: string): Promise<{ filename: string; text: string; category: string }[]> {
  const files = fs.readdirSync(dir).filter(file => file.endsWith('.pdf'));
  const results = [];

  for (const file of files) {
    const filePath = path.join(dir, file);
    const dataBuffer = fs.readFileSync(filePath);
    const pdfData = await pdf(dataBuffer);

    const category = pdf_category_mapping[file] || "Uncategorized";

    results.push({
      filename: file,
      text: pdfData.text,
      category
    });
  }

  return results;
}

async function loadData() {
  const collection = await astraDb.collection(COLLECTION_NAME);

  const pdfDir = './pdfs';
  const pdfDocs = await loadPDFsFromDirectory(pdfDir);

  for (const doc of pdfDocs) {
    const chunks = await splitter.splitText(doc.text);
    let i = 0;
    for await (const chunk of chunks) {
      const { data } = await openai.embeddings.create({ input: chunk, model: 'text-embedding-3-small' });
      await collection.insertOne({
        id: uuidv4(),
        document_id: doc.filename,
        chunk_id: `${doc.filename}-${i}`,
        text: chunk,
        category: doc.category,
        $vector: data[0]?.embedding
      });
      i++;
    }
  }

  console.log('All PDF data loaded');
}

async function run() {
  await createCollection();
  await loadData();
}

run().catch(console.error);
