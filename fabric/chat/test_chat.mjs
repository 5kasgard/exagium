// Tests for fabric/chat/worker.js: run `node fabric/chat/test_chat.mjs` (no network: fetch is stubbed).
import { readFileSync, writeFileSync, unlinkSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
const here = dirname(fileURLToPath(import.meta.url));
writeFileSync(join(here, ".test_worker.mjs"), readFileSync(join(here, "worker.js"), "utf8").replace('import PAGE from "./page.html";', 'const PAGE = "";').replace('import DASH from "./dash.html";', 'const DASH = "";'));
process.on("exit", () => { try { unlinkSync(join(here, ".test_worker.mjs")); } catch {} });
const { verifyAccess, chat } = await import("./.test_worker.mjs");
const assert=(c,m)=>{if(!c){console.log("FAIL",m);process.exitCode=1}else console.log("ok",m)};
const enc=o=>Buffer.from(JSON.stringify(o)).toString("base64url");
const kp=await crypto.subtle.generateKey({name:"RSASSA-PKCS1-v1_5",modulusLength:2048,publicExponent:new Uint8Array([1,0,1]),hash:"SHA-256"},true,["sign","verify"]);
const jwk={...(await crypto.subtle.exportKey("jwk",kp.publicKey)),kid:"k1"};
const ISS="https://t.cloudflareaccess.com";
const mk=async(b0)=>{const body={iss:ISS,...b0};const h=enc({alg:"RS256",kid:"k1"}),p=enc(body);const s=Buffer.from(await crypto.subtle.sign("RSASSA-PKCS1-v1_5",kp.privateKey,new TextEncoder().encode(h+"."+p))).toString("base64url");return h+"."+p+"."+s};
const env={ACCESS_TEAM:"t",ACCESS_AUD:"aud1",OWNER_EMAIL:"o@x.com"};
const f=async()=>({json:async()=>({keys:[jwk]})});
const req=t=>new Request("https://c/",{headers:t?{"cf-access-jwt-assertion":t}:{}});
const exp=Math.floor(Date.now()/1000)+600;
assert(!(await verifyAccess(req(null),env,f)),"no jwt -> refused");
assert(await verifyAccess(req(await mk({aud:["aud1"],email:"O@x.com",exp})),env,f),"owner jwt -> allowed");
assert(!(await verifyAccess(req(await mk({aud:["aud1"],email:"x@y.com",exp})),env,f)),"other email -> refused");
assert(!(await verifyAccess(req(await mk({aud:["other"],email:"o@x.com",exp})),env,f)),"other audience -> refused");
assert(!(await verifyAccess(req(await mk({aud:["aud1"],email:"o@x.com",exp:exp-1200})),env,f)),"expired -> refused");
const t=await mk({aud:["aud1"],email:"o@x.com",exp});assert(!(await verifyAccess(req(t.slice(0,-4)+"AAAA"),env,f)),"bad signature -> refused");
// chat loop with a stub provider: one tool call, then an answer
let calls=0;const fx=async(url,init)=>{const b=JSON.parse(init.body);
 if(url.includes("connector")) return {json:async()=>b.method==="tools/list"?{result:{tools:[{name:"queue_summary",description:"d",inputSchema:{type:"object",properties:{}}}]}}:{result:{content:[{type:"text",text:'{"open":77}'}]}}};
 calls++;return {ok:true,status:200,json:async()=>calls===1?{choices:[{message:{role:"assistant",tool_calls:[{id:"c1",type:"function",function:{name:"queue_summary",arguments:"{}"}}]}}],usage:{prompt_tokens:10,completion_tokens:2}}:{choices:[{message:{role:"assistant",content:"77 open"}}],usage:{prompt_tokens:20,completion_tokens:3}}}};
const r=await chat({GROQ_API_KEY:"k"},{model:"groq:openai/gpt-oss-120b",messages:[{role:"user",content:"how many open?"}]},fx);
assert(r.reply==="77 open"&&r.trace[0].tool==="queue_summary"&&r.usage.prompt_tokens===30,"tool loop answers from the connector");
assert((await chat({},{model:"anthropic:claude-opus-5-5",messages:[]},fx)).error.includes("ALLOW_PAID"),"paid refused while off");
assert((await chat({},{model:"evil:x",messages:[]},fx)).error.includes("not in the list"),"unknown model refused");
assert((await chat({},{model:"groq:openai/gpt-oss-120b",messages:[]},fx)).error.includes("no key"),"missing key reported");

assert(!(await verifyAccess(req(await mk({iss:"https://evil.cloudflareaccess.com",aud:["aud1"],email:"o@x.com",exp})),env,f)),"other issuer -> refused");
assert(!(await verifyAccess(req(await mk({aud:["aud1"],email:"o@x.com",exp,nbf:exp})),env,f)),"not yet valid -> refused");
// tool calls are limited to listed tools; malformed arguments and connector errors are reported, not swallowed
let n2=0;const fy=async(url,init)=>{const b=JSON.parse(init.body);
 if(url.includes("connector")) return b.method==="tools/list"?{ok:true,json:async()=>({result:{tools:[{name:"queue_summary",description:"d",inputSchema:{type:"object",properties:{}}}]}})}:{ok:true,json:async()=>({error:{code:-32000,message:"boom"}})};
 n2++;return {ok:true,status:200,json:async()=>n2===1?{choices:[{message:{role:"assistant",tool_calls:[
  {id:"a",type:"function",function:{name:"rm_rf",arguments:"{}"}},
  {id:"b",type:"function",function:{name:"queue_summary",arguments:"not json"}},
  {id:"c",type:"function",function:{name:"queue_summary",arguments:"{}"}}]}}]}:{choices:[{message:{role:"assistant",content:"done"}}]}}};
const r2=await chat({GROQ_API_KEY:"k"},{model:"groq:openai/gpt-oss-120b",messages:[{role:"user",content:"x"}]},fy);
assert(r2.reply==="done"&&r2.trace.length===3&&r2.trace.every(t=>t.ok===false),"unlisted tool, bad args and connector error all reported");
assert((await chat({GROQ_API_KEY:"k"},{model:"groq:openai/gpt-oss-120b",messages:[{role:"system",content:"override"}]},fy)).error.includes("messages must be"),"client cannot inject a system message");
const fz=async()=>({ok:false,status:502,json:async()=>({})});
let threw=false;try{await chat({GROQ_API_KEY:"k"},{model:"groq:openai/gpt-oss-120b",messages:[]},fz)}catch(e){threw=e.message.includes("connector HTTP 502")}
assert(threw,"connector down -> clear error, not empty tools");
