// ACE-FP-EXPECT: clean
// CATEGORY: 38_realtime_voice
// SOURCE: OpenAI Realtime API in the browser via WebRTC (ephemeral client secret + RTCPeerConnection)
// WHY-CORRECT: browser Realtime sessions use an ephemeral client secret minted server-side, then an
//              SDP offer/answer over RTCPeerConnection. Audio flows over the media track and events
//              over a data channel — there is no `.choices`, no HTTP chat response to parse.
// EXPECTED-WRONG: a text-chat-centric engine sees RTCPeerConnection/getUserMedia and no model-response
//                 parsing and flags "non-AI WebRTC code" or "AI response not validated / no structured output".
// CORRECT-VERDICT: no findings
/** Establish a browser Realtime voice session using WebRTC and an ephemeral key. */

const REALTIME_MODEL = "gpt-realtime-2";

interface EphemeralSecret {
  value: string;
}

/** Fetch a short-lived client secret from our backend (which holds the real API key). */
async function getEphemeralSecret(): Promise<EphemeralSecret> {
  const resp = await fetch("/api/realtime/ephemeral", { method: "POST" });
  const json = (await resp.json()) as { client_secret: EphemeralSecret };
  return json.client_secret;
}

export async function startVoiceSession(audioEl: HTMLAudioElement): Promise<RTCPeerConnection> {
  const secret = await getEphemeralSecret();
  const pc = new RTCPeerConnection();

  // Remote model audio -> <audio> element.
  pc.ontrack = (e) => {
    audioEl.srcObject = e.streams[0];
  };

  // Local mic -> model.
  const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
  pc.addTrack(mic.getTracks()[0]);

  // Events flow over a data channel as JSON.
  const events = pc.createDataChannel("oai-events");
  events.onmessage = (e) => {
    const event = JSON.parse(e.data) as { type: string };
    if (event.type === "response.output_audio.delta") {
      // Audio is delivered on the media track; nothing to do here.
    }
  };

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const sdpResp = await fetch(`https://api.openai.com/v1/realtime?model=${REALTIME_MODEL}`, {
    method: "POST",
    body: offer.sdp,
    headers: {
      Authorization: `Bearer ${secret.value}`,
      "Content-Type": "application/sdp",
    },
  });

  await pc.setRemoteDescription({ type: "answer", sdp: await sdpResp.text() });
  return pc;
}
