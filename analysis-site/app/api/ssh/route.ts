import { NextRequest, NextResponse } from 'next/server';
import { NodeSSH } from 'node-ssh';

export async function POST(req: NextRequest) {
  try {
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
    
    const result = await ssh.execCommand('ls');
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