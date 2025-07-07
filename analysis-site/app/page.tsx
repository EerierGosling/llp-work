import Image from "next/image";

import { NextRequest, NextResponse } from 'next/server';
import { NodeSSH } from 'node-ssh';
import { writeFile } from 'fs/promises';

async function runAnalysis(formData: FormData) {
  'use server';

  try {

    const file = formData.get('file') as File;

    const ssh = new NodeSSH();
    
    await ssh.connect({
      host: 'neuronic.cs.princeton.edu',
      username: 'se0361',
      password: process.env.SSH_PASSWORD || '',
      tryKeyboard: true,
      onKeyboardInteractive(name, instructions, lang, prompts, finish) {            
        if (prompts.length > 0) {
          const prompt = prompts[0].prompt.toLowerCase();
          
          if (prompt.includes('password')) {
            finish([process.env.SSH_PASSWORD || '']);
          }
          else {
            finish(['1']);
          }
        }
      }
    });
    
    await new Promise(resolve => setTimeout(resolve, 2000));

    await ssh.execCommand('cd /n/fs/visualai-scr/temp_LLP/sofia/');

    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const tempPath = join('/website-images', file.name);
    await writeFile(tempPath, buffer);

    const result = await ssh.execCommand(`sbatch --nodes=1 --gres=gpu:1 --mem=50G -t 01:00:00 --wrap=\"source ~/.bashrc && conda activate sofia && python analysis.py --file_name ${file.name}\"`);

    ssh.dispose();
    
    return NextResponse.json({ success: true, output: result.stdout });
  } catch (error) {
    console.error('SSH operation failed:', error);
    return NextResponse.json(
      { success: false, error: (error as Error).message },
      { status: 500 }
    );
  }
}

export default function Home() {
  return (
    
  );
}
