export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const gameUrl = `${url.origin}/game`;

    // 0) Telegram Mini App: Brain & Memory Games at /game
    if (request.method === "GET" && url.pathname === "/game") {
      return new Response(getBrainGameHtml(), {
        status: 200,
        headers: {
          "Content-Type": "text/html; charset=utf-8",
          "Cache-Control": "no-store",
        },
      });
    }

    // 1) One-click webhook setup route
    if (request.method === "GET" && url.pathname === "/setup") {
      if (!env.TELEGRAM_BOT_TOKEN) {
        return new Response("Missing TELEGRAM_BOT_TOKEN secret in Cloudflare.", { status: 500 });
      }
      const webhookUrl = `${url.origin}/webhook`;
      const tgResp = await fetch(
        `https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/setWebhook`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: webhookUrl, drop_pending_updates: true }),
        }
      );
      const data = await tgResp.json();
      return new Response(
        JSON.stringify(
          {
            webhookUrl,
            gameUrl,
            gemini_keys_configured: getGeminiKeys(env).length,
            groq_configured: Boolean(env.GROQ_API_KEY),
            workers_ai_bound: Boolean(env.AI),
            kv_memory_bound: Boolean(env.CHAT_MEMORY),
            telegram: data,
          },
          null,
          2
        ),
        { headers: { "Content-Type": "application/json; charset=utf-8" } }
      );
    }

    if (request.method === "GET") {
      return new Response("Iran Bioinformatics Telegram Agent Webhook is running.", { status: 200 });
    }

    if (request.method !== "POST") {
      return new Response("Method Not Allowed", { status: 405 });
    }

    let chatId = "";
    try {
      const update = await request.json();
      const message = update.message || update.edited_message;
      const incomingAudio = extractAudioInfo(message);
      const rawMsgText = (message?.text || message?.caption || "").trim();
      if (!message || (!rawMsgText && !incomingAudio)) {
        return new Response("OK", { status: 200 });
      }

      chatId = String(message.chat.id);
      const fromId = message.from?.id ? String(message.from.id) : "";
      const isGroup = message.chat.type === "group" || message.chat.type === "supergroup";
      const rawAllowed = String(env.ALLOWED_CHAT_ID || "").trim();
      const allowedList = rawAllowed.split(",").map((s) => s.trim()).filter(Boolean);
      const text = rawMsgText;

      // Owner-only /id helper
      if (/^\/id(@\w+)?$/i.test(text)) {
        if (!rawAllowed || rawAllowed === "*" || allowedList.includes(fromId) || allowedList.includes(chatId)) {
          const suggested = allowedList.includes(chatId)
            ? rawAllowed
            : [...new Set([...allowedList, chatId])].join(",");
          await sendTelegram(
            env.TELEGRAM_BOT_TOKEN,
            chatId,
            `🆔 آیدی این چت: <code>${chatId}</code>\n\nبرای فعال شدن ربات در این گروه، مقدار <b>ALLOWED_CHAT_ID</b> را در کلودفلر به این صورت بگذارید:\n<code>${escapeHtml(suggested)}</code>`,
            null,
            message.message_id
          );
        }
        return new Response("OK", { status: 200 });
      }

      if (rawAllowed && rawAllowed !== "*" && !allowedList.includes(chatId)) {
        if (!isGroup) {
          await sendTelegram(env.TELEGRAM_BOT_TOKEN, chatId, "⛔ این ربات خصوصی است.");
        }
        return new Response("OK", { status: 200 });
      }

      // Group gating: respond only if commanded, replied to bot, mentioned, or starts with bot trigger word
      if (isGroup) {
        const hasCmd = text.startsWith("/");
        const isReplyToBot = Boolean(message.reply_to_message?.from?.is_bot);
        const hasMention = (message.entities || []).some((e) => e.type === "mention");
        const startsBotWord = /^(ربات|بات|بازی|سرگرمی|خبر|اخبار|خبرهای|امروز|آموزش|پست)(?=\s|$|[!؟،.])/i.test(text);
        if (!hasCmd && !isReplyToBot && !hasMention && !startsBotWord) {
          return new Response("OK", { status: 200 });
        }
        if (hasMention && !hasCmd && !isReplyToBot) {
          const meResp = await fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/getMe`);
          const meData = meResp.ok ? await meResp.json() : null;
          const botUser = meData?.result?.username ? `@${meData.result.username.toLowerCase()}` : "";
          if (botUser && !text.toLowerCase().includes(botUser)) {
            return new Response("OK", { status: 200 });
          }
        }
      }

      const forceNews = /^\/(news|خبر)(?=\s|$)/i.test(text);
      const forceEdu = /^\/(edu|آموزش)(?=\s|$)/i.test(text);

      const cleanText = text
        .replace(/@[A-Za-z0-9_]+/g, "")
        .replace(/^\/(news|edu|ask|خبر|آموزش|new|reset)(?=\s|$)/i, "")
        .replace(/^(ربات|بات)(?=\s|$)\s*/i, "")
        .trim();

      const gameMarkup = {
        inline_keyboard: [
          [
            isGroup
              ? { text: "🧠 بازی تقویت حافظه تا آماده شدن خبرها", url: gameUrl }
              : { text: "🧠 بازی تقویت حافظه تا آماده شدن خبرها", web_app: { url: gameUrl } },
          ],
        ],
      };

      // Reset conversation memory
      if (/^\/(new|reset)(@\w+)?$/i.test(text) || /^(موضوع جدید|بحث جدید|شروع جدید|چت جدید)[!؟.\s]*$/i.test(cleanText)) {
        await clearChatMemory(env, chatId);
        await sendTelegram(
          env.TELEGRAM_BOT_TOKEN,
          chatId,
          "✨ گفت‌وگوی جدید باز شد!\nموضوع یا ایده‌ای که می‌خوای رویش کار کنیم رو بفرست تا با هم بسازیمش.",
          null,
          isGroup ? message.message_id : null
        );
        return new Response("OK", { status: 200 });
      }

      // Mini-game shortcut
      if (/^\/game(@\w+)?$/i.test(text) || cleanText === "بازی" || cleanText === "سرگرمی" || cleanText === "بازی فکری") {
        await sendTelegram(
          env.TELEGRAM_BOT_TOKEN,
          chatId,
          "🧠 <b>باشگاه تقویت حافظه و تمرکز آکادمی</b>\n\nدکمه زیر را بزنید تا ۳ بازی فکری (ماتریس حافظه، دنباله DNA و تست سرعت رنگ) باز شود:",
          gameMarkup,
          isGroup ? message.message_id : null
        );
        return new Response("OK", { status: 200 });
      }

      // GitHub Actions live status diagnostic (/status)
      if (/^\/status(@\w+)?$/i.test(text) || /^(وضعیت|وضعیت سرور|چرا گیر کرد|تست سرور)[!؟.\s]*$/i.test(cleanText)) {
        await sendTyping(env.TELEGRAM_BOT_TOKEN, chatId);
        const statusHtml = await checkGitHubActionsStatus(env);
        await sendTelegram(env.TELEGRAM_BOT_TOKEN, chatId, statusHtml, null, isGroup ? message.message_id : null);
        return new Response("OK", { status: 200 });
      }

      // Live Academy Course Knowledge Commands (/course, /setcourse, /clearcourse)
      if (/^\/course(@\w+)?$/i.test(text) || /^(لیست\s*سرفصل|سرفصل[‌ ]?های\s*دوره|دوره[‌ ]?های\s*آکادمی)[!؟.\s]*$/i.test(cleanText)) {
        await sendTyping(env.TELEGRAM_BOT_TOKEN, chatId);
        const kb = await fetchAcademyCourses(env);
        const summaryHtml = formatAcademyOverviewHtml(kb);
        await sendTelegram(env.TELEGRAM_BOT_TOKEN, chatId, summaryHtml, null, isGroup ? message.message_id : null);
        return new Response("OK", { status: 200 });
      }

      if (/^\/clearcourse(@\w+)?$/i.test(text)) {
        if (env.CHAT_MEMORY) {
          await env.CHAT_MEMORY.delete("kb:academy_courses_custom");
          await env.CHAT_MEMORY.delete("kb:academy_courses_v1");
        }
        await sendTelegram(env.TELEGRAM_BOT_TOKEN, chatId, "🧹 یادداشت‌های سفارشی دوره پاک شد و دانش پایه از گیت‌هاب مجدداً بارگذاری می‌شود.", null, isGroup ? message.message_id : null);
        return new Response("OK", { status: 200 });
      }

      const setCourseMatch = text.match(/^\/setcourse(?:@\w+)?(?:\s+([\s\S]+))?$/i) ||
        cleanText.match(/^(?:ذخیره\s*اطلاعات\s*دوره|به\s*اطلاعات\s*دوره\s*اضافه\s*کن|آپدیت\s*اطلاعات\s*دوره)[:\s\n]*([\s\S]*)$/i);
      if (setCourseMatch) {
        const inlineNote = (setCourseMatch[1] || "").trim();
        const repliedNote = (message.reply_to_message?.text || message.reply_to_message?.caption || "").trim();
        const noteToSave = inlineNote || repliedNote;
        if (!noteToSave || noteToSave.length < 10) {
          await sendTelegram(
            env.TELEGRAM_BOT_TOKEN,
            chatId,
            "📝 برای ثبت اطلاعات جدید یا تکمیلی دوره‌ها (مثل دوره سرطان، تاریخ شروع، کد تخفیف یا تغییر استاد)، پیام را بنویسید یا روی پیام/رونوشت ویس ریپلای بزنید و بنویسید:\n<code>/setcourse متن اطلاعات جدید...</code>",
            null,
            isGroup ? message.message_id : null
          );
          return new Response("OK", { status: 200 });
        }
        const savedCount = await appendCustomCourseNote(env, noteToSave);
        await sendTelegram(
          env.TELEGRAM_BOT_TOKEN,
          chatId,
          `✅ <b>اطلاعات جدید دوره در حافظه آکادمی ذخیره شد!</b>\n📚 تعداد یادداشت‌های تکمیلی فعال: <b>${toPersianDigits(savedCount)}</b>\n\nاز این لحظه در تولید محتوا، پاسخ‌گویی و ساخت اسلایدها این اطلاعات هم لحاظ می‌شود.`,
          null,
          isGroup ? message.message_id : null
        );
        return new Response("OK", { status: 200 });
      }

      // Help / Start
      if (/^\/(start|help)(@\w+)?$/i.test(text) || cleanText === "سلام" || cleanText === "راهنما") {
        const helpMsg =
          "👋 سلام! اینجا همکار محتواسازی و اخبار <b>آکادمی بیوانفورماتیک ایران</b> است:\n\n" +
          "💬 <b>گفت‌وگو و ساخت محتوای آموزشی (فوری در ۲ ثانیه):</b>\n" +
          "همین‌جا راحت با من صحبت کن، ایده بخواه، اصلاح بده («کوتاه‌ترش کن»، «این تیکه رو بازتر بگو»، «کپشن اینستا بده»، «منابع گیت‌هابشو بده»، «با امضا نهاییش کن»).\n" +
          "• برای شروع بحث تازه: <code>/new</code> یا <b>موضوع جدید</b>\n\n" +
          "🎓 <b>آشنایی کامل با دوره‌ها و اساتید آکادمی:</b>\n" +
          "کل سرفصل‌ها، ساعات، ابزارها و اساتید طرح ملی بیوانفورماتیک را بلدم! کافی بگی: «طبق سرفصل Bulk RNA-Seq دوره خودمون یه پست بساز» یا «ترکیبی از دوره ما و منابع گیت‌هاب بنویس».\n" +
          "• مشاهده فهرست کل سرفصل‌ها: <code>/course</code> | افزودن اطلاعات جدید دوره: <code>/setcourse</code>\n\n" +
          "🎙 <b>تبدیل ویس و صوت هر زبانی به متن دقیق (رایگان + ۳ موتور هوشمند):</b>\n" +
          "کافیه هر ویس، فایل صوتی یا ویدئو-نوت به هر زبانی (فارسی، انگلیسی، دو‌زبانه و...) بفرستی تا دقیق و بدون خطا تایپ کنم (حتی می‌تونی بعدش بگی همین رو خلاصه یا ویرایش کن!).\n\n" +
          "📰 <b>اخبار علمی زنده (رصد ۴۰ پایگاه + طراحی تصویر اختصاصی خبر):</b>\n" +
          "بگو: <b>۵ خبر از بیوانفورماتیک بفرست</b> یا <b>/news پاروویروس</b>\n\n" +
          "🧠 بازی فکری: <code>/game</code> | 📡 وضعیت سرور: <code>/status</code>";
        await sendTelegram(env.TELEGRAM_BOT_TOKEN, chatId, helpMsg, gameMarkup, isGroup ? message.message_id : null);
        return new Response("OK", { status: 200 });
      }

      const rawReqText = cleanText || text;

      // Inspect reply_to_message (ignoring status/help messages)
      const replyMsg = message.reply_to_message;
      const replyAudio = extractAudioInfo(replyMsg);
      const replyRawText = (replyMsg?.text || replyMsg?.caption || "").trim();
      const isStatusOrHelp = /^(⏳|✅|🎓 درخواست|📰 درخواست|👋 سلام|🧠|🤖 برای خبر|✨ گفت‌وگو|⛔|🎙 رونوشت)/.test(replyRawText);
      const isContentReply = Boolean(replyRawText && replyRawText.length >= 25 && !isStatusOrHelp);

      // 0.3) Triple-Engine Beast-Mode Voice & Audio Transcription (Groq Whisper v3 + Gemini Audio + CF Workers AI Whisper)
      const activeAudio = incomingAudio || (replyAudio && /(تایپ|متن|بنویس|رونوشت|ترجمه|خلاصه|تبدیل|transcribe|text)/i.test(rawReqText) ? replyAudio : null);
      if (activeAudio) {
        await sendTyping(env.TELEGRAM_BOT_TOKEN, chatId);
        const memVoice = await loadChatMemory(env, chatId);
        const res = await transcribeVoiceBeastMode(env, activeAudio);

        if (!res.ok) {
          await sendTelegram(
            env.TELEGRAM_BOT_TOKEN,
            chatId,
            `⚠️ خطا در تبدیل صدا به متن:\n<code>${escapeHtml(res.error || "خطای نامشخص")}</code>`,
            null,
            isGroup ? message.message_id : null
          );
          return new Response("OK", { status: 200 });
        }

        const langTag = res.language ? ` | 🌍 زبان: <b>${escapeHtml(res.language)}</b>` : "";
        const durTag = res.durationSec ? ` | ⏱ ${toPersianDigits(Math.round(res.durationSec))} ثانیه` : "";
        const engTag = res.enginesUsed?.length ? `\n⚙️ موتور: <code>${escapeHtml(res.enginesUsed.join(" + "))}</code>` : "";
        const headerHtml = `🎙 <b>رونوشت دقیق ویس</b>${langTag}${durTag}${engTag}\n\n`;

        await sendTelegram(
          env.TELEGRAM_BOT_TOKEN,
          chatId,
          `${headerHtml}${escapeHtml(res.transcript)}`,
          null,
          isGroup ? message.message_id : null
        );

        const updatedHist = [
          ...memVoice.history,
          { role: "user", text: `[ویس کاربر — ${res.language || "خودکار"}]: ${res.transcript.slice(0, 1200)}${rawReqText ? `\n[دستور همراه ویس]: ${rawReqText}` : ""}` },
          { role: "model", text: res.transcript },
        ].slice(-14);
        await saveChatMemory(env, chatId, updatedHist, res.transcript);

        // If the user also included a follow-up instruction on the voice caption or reply (beyond "transcribe")
        const wantsOnlyTranscribe = !rawReqText || /^(تایپ|تایپش\s*کن|متن|متنش\s*رو\s*بده|متنشو\s*بده|بنویس|رونوشت|transcribe)[!؟.\s]*$/i.test(rawReqText);
        if (!wantsOnlyTranscribe) {
          const kbVoice = await fetchAcademyCourses(env);
          const libVoice = await fetchBioinfoLibrary(env);
          const resolvedVoice = resolveKnowledgeSources({
            queryText: rawReqText,
            contextSeed: res.transcript.slice(0, 600),
            kb: kbVoice,
            lib: libVoice,
          });
          await sendTyping(env.TELEGRAM_BOT_TOKEN, chatId);
          const extraOut = await runConversationalCollab({
            env,
            userText: rawReqText,
            history: updatedHist,
            repliedMessage: res.transcript.slice(0, 3500),
            matchedRepos: resolvedVoice.matchedRepos,
            matchedCourseContext: resolvedVoice.courseContext,
            knowledgeModeTag: resolvedVoice.modeTag,
          });
          await sendTelegram(env.TELEGRAM_BOT_TOKEN, chatId, extraOut, null, isGroup ? message.message_id : null);
          const nextHist = [...updatedHist, { role: "model", text: extraOut }].slice(-14);
          await saveChatMemory(env, chatId, nextHist, extraOut.length >= 120 ? extraOut : res.transcript);
        }
        return new Response("OK", { status: 200 });
      }

      const mem = await loadChatMemory(env, chatId);

      // 1) Decide if this goes to GitHub Actions (ONLY News requests — News has image generation)
      const isNewsRequest = !forceEdu && isExplicitNewsIntent(text, rawReqText, forceNews, isContentReply);

      if (isNewsRequest) {
        const parsed = await parseNewsRequest(rawReqText, env);
        const topicLabel = parsed.persian_topic || parsed.english_topic || "حوزه‌های منتخب";
        const reqDesc = `📰 درخواست <b>${toPersianDigits(parsed.count)} خبر</b> درباره <b>${escapeHtml(topicLabel)}</b> ثبت شد.`;

        const statusResp = await sendTelegram(
          env.TELEGRAM_BOT_TOKEN,
          chatId,
          `${reqDesc}\n\n<code>[▓░░░░░░░░░] ۱۰٪</code>\nدر حال بیدار کردن سرور پردازش...`,
          gameMarkup,
          isGroup ? message.message_id : null
        );
        const statusMsgId = statusResp?.result?.message_id ? String(statusResp.result.message_id) : "";

        const triggered = await triggerGitHubWorkflow(env, {
          topic: parsed.english_topic,
          count: String(parsed.count),
          custom_query: rawReqText,
          on_demand: "true",
          reply_chat_id: chatId,
          status_message_id: statusMsgId,
          game_url: gameUrl,
          mode: "news",
          edu_images: "false",
          reply_context: "",
          aspect_ratio: "post",
          design_spec: "",
        });

        await finalizeDispatchStatus(env.TELEGRAM_BOT_TOKEN, chatId, statusMsgId, reqDesc, triggered, gameMarkup);
        return new Response("OK", { status: 200 });
      }

      // 2) Instant 2-Second Conversational Collaborator (Chat + Drafting + Editing + Course/Library Sources)
      await sendTyping(env.TELEGRAM_BOT_TOKEN, chatId);
      const contextSeed = isContentReply ? replyRawText.slice(0, 400) : (mem.last_draft || "").slice(0, 350);

      const [kbCourses, libCourses] = await Promise.all([
        fetchAcademyCourses(env),
        /(منابع|منبع|ریپو|ریپوها|گیت[‌ ]?هاب|ابزار|لینک|کتابخانه|سایر|بقیه|خارج|تکمیلی|ترکیب|هر\s*دو|هردو|هم\s+.*\s+هم)/i.test(rawReqText)
          ? fetchBioinfoLibrary(env)
          : Promise.resolve(null),
      ]);

      const resolvedKnowledge = resolveKnowledgeSources({
        queryText: rawReqText,
        contextSeed,
        kb: kbCourses,
        lib: libCourses,
      });

      const replyTextOut = await runConversationalCollab({
        env,
        userText: rawReqText,
        history: mem.history,
        repliedMessage: isContentReply ? replyRawText.slice(0, 3500) : "",
        matchedRepos: resolvedKnowledge.matchedRepos,
        matchedCourseContext: resolvedKnowledge.courseContext,
        knowledgeModeTag: resolvedKnowledge.modeTag,
      });

      await sendTelegram(
        env.TELEGRAM_BOT_TOKEN,
        chatId,
        replyTextOut,
        null,
        isGroup ? message.message_id : null
      );

      // Persist conversation memory in Cloudflare KV
      const userEntryText = isContentReply
        ? `[در پاسخ به پیام: ${replyRawText.slice(0, 500)}]\n${rawReqText}`
        : rawReqText;
      const updatedHistory = [
        ...mem.history,
        { role: "user", text: userEntryText },
        { role: "model", text: replyTextOut },
      ].slice(-14);
      const nextDraft = replyTextOut.length >= 120 ? replyTextOut : mem.last_draft;
      await saveChatMemory(env, chatId, updatedHistory, nextDraft);

      return new Response("OK", { status: 200 });
    } catch (err) {
      if (chatId && env.TELEGRAM_BOT_TOKEN) {
        try {
          await sendTelegram(
            env.TELEGRAM_BOT_TOKEN,
            chatId,
            `⚠️ خطای موقت در پردازش پیام:\n<code>${escapeHtml(err?.message || String(err))}</code>`
          );
        } catch (_) {}
      }
      return new Response(`Error: ${err.message}`, { status: 200 });
    }
  },
};

function toEnglishDigits(str) {
  const map = { "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4", "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9" };
  return String(str || "").replace(/[۰-۹]/g, (d) => map[d] || d);
}

function toPersianDigits(val) {
  const map = ["۰", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"];
  return String(val).replace(/\d/g, (d) => map[Number(d)]);
}

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function getGeminiKeys(env) {
  const raw = String(env.GEMINI_API_KEY || "").trim();
  if (!raw) return [];
  const parts = raw.split(/[,;\s\n]+/).map((s) => s.trim()).filter(Boolean);
  return [...new Set(parts)];
}

function isExplicitNewsIntent(fullText, cleanText, forceNews, isContentReply) {
  if (forceNews) return true;
  if (isContentReply) return false; // Replies refine/discuss the replied post instead of running news search
  const t = `${fullText || ""} ${cleanText || ""}`.trim();
  const hasNewsWord = /(?:^|\s)(خبر|اخبار|خبرهای|خبرها|تازه‌های\s+خبر|جدیدترین\s+خبر|news)(?=\s|$|[!؟،.])/i.test(t);
  const hasCreateEdu = /(پست|محتوا|اسلاید|کاروسل|آموزش|بنویس|کپشن)/i.test(t);
  if (hasNewsWord && !hasCreateEdu) return true;
  // Also catch "۵ تا خبر درباره آموزش X"
  if (/^(?:\d+|[۰-۹]+)?\s*(تا\s+)?(خبر|اخبار|خبرهای)/i.test(cleanText || "")) return true;
  return false;
}

function arrayBufferToBase64(buf) {
  const bytes = new Uint8Array(buf);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function extractAudioInfo(msg) {
  if (!msg) return null;
  if (msg.voice?.file_id) {
    return {
      fileId: msg.voice.file_id,
      mimeType: "audio/ogg",
      ext: "ogg",
      duration: msg.voice.duration || 0,
      fileSize: msg.voice.file_size || 0,
    };
  }
  if (msg.audio?.file_id) {
    const mime = String(msg.audio.mime_type || "audio/mpeg");
    const ext = mime.includes("mp4") || mime.includes("m4a") ? "m4a" : (mime.includes("ogg") ? "ogg" : (mime.includes("wav") ? "wav" : (mime.includes("flac") ? "flac" : "mp3")));
    return {
      fileId: msg.audio.file_id,
      mimeType: mime === "audio/x-m4a" ? "audio/mp4" : mime,
      ext,
      duration: msg.audio.duration || 0,
      fileSize: msg.audio.file_size || 0,
    };
  }
  if (msg.video_note?.file_id) {
    return {
      fileId: msg.video_note.file_id,
      mimeType: "video/mp4",
      ext: "mp4",
      duration: msg.video_note.duration || 0,
      fileSize: msg.video_note.file_size || 0,
    };
  }
  if (msg.document?.file_id) {
    const mime = String(msg.document.mime_type || "").toLowerCase();
    const fname = String(msg.document.file_name || "").toLowerCase();
    if (mime.startsWith("audio/") || /\.(ogg|oga|opus|mp3|m4a|wav|flac|webm)$/.test(fname)) {
      const ext = (fname.match(/\.([a-z0-9]+)$/)?.[1]) || "ogg";
      const normMime = ext === "oga" || ext === "ogg" || ext === "opus" ? "audio/ogg" : (ext === "m4a" ? "audio/mp4" : (mime || "audio/mpeg"));
      return {
        fileId: msg.document.file_id,
        mimeType: normMime,
        ext: ext === "oga" ? "ogg" : ext,
        duration: 0,
        fileSize: msg.document.file_size || 0,
      };
    }
  }
  return null;
}

async function transcribeVoiceBeastMode(env, audioInfo) {
  if (!audioInfo?.fileId || !env.TELEGRAM_BOT_TOKEN) {
    return { ok: false, error: "فایل صوتی معتبر پیدا نشد." };
  }
  if (audioInfo.fileSize && audioInfo.fileSize > 20 * 1024 * 1024) {
    return { ok: false, error: "حجم فایل بیشتر از سقف ۲۰ مگابایت تلگرام است؛ لطفاً با حجم کمتر یا در چند بخش بفرستید." };
  }

  let audioBuf = null;
  let ext = audioInfo.ext || "ogg";
  let mimeType = audioInfo.mimeType || "audio/ogg";
  try {
    const gf = await fetch(
      `https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/getFile?file_id=${encodeURIComponent(audioInfo.fileId)}`,
      { signal: AbortSignal.timeout(8000) }
    );
    const gfData = gf.ok ? await gf.json() : null;
    const fp = gfData?.result?.file_path;
    if (!fp) {
      return { ok: false, error: gfData?.description || "دریافت مسیر فایل از تلگرام ناموفق بود (ممکن است فایل بالای ۲۰ مگابایت باشد)." };
    }
    if (fp.endsWith(".oga") || fp.endsWith(".ogg")) { ext = "ogg"; mimeType = "audio/ogg"; }
    else if (fp.endsWith(".mp3")) { ext = "mp3"; mimeType = "audio/mpeg"; }
    else if (fp.endsWith(".m4a")) { ext = "m4a"; mimeType = "audio/mp4"; }
    else if (fp.endsWith(".wav")) { ext = "wav"; mimeType = "audio/wav"; }
    else if (fp.endsWith(".mp4")) { ext = "mp4"; mimeType = "video/mp4"; }

    const dl = await fetch(`https://api.telegram.org/file/bot${env.TELEGRAM_BOT_TOKEN}/${fp}`, {
      signal: AbortSignal.timeout(18000),
    });
    if (!dl.ok) return { ok: false, error: `دانلود فایل صوتی ناموفق بود (${dl.status}).` };
    audioBuf = await dl.arrayBuffer();
  } catch (e) {
    return { ok: false, error: `خطا در دانلود فایل صوتی: ${e?.message || String(e)}` };
  }

  const enginesUsed = [];
  const errors = [];
  let whisperText = "";
  let detectedLang = "";
  let durationSec = audioInfo.duration || 0;

  // Engine 1: Groq Whisper Large v3 (with whisper-large-v3-turbo fallback) — 132x realtime, up to 25MB binary FormData
  const groqKey = String(env.GROQ_API_KEY || "").trim();
  if (groqKey) {
    for (const wModel of ["whisper-large-v3", "whisper-large-v3-turbo"]) {
      try {
        const fd = new FormData();
        fd.append("file", new Blob([audioBuf], { type: mimeType }), `voice.${ext}`);
        fd.append("model", wModel);
        fd.append("temperature", "0");
        fd.append("response_format", "verbose_json");
        const r = await fetch("https://api.groq.com/openai/v1/audio/transcriptions", {
          method: "POST",
          headers: { Authorization: `Bearer ${groqKey}` },
          body: fd,
          signal: AbortSignal.timeout(22000),
        });
        if (r.ok) {
          const d = await r.json();
          const txt = String(d?.text || "").trim();
          if (txt) {
            whisperText = txt;
            detectedLang = String(d?.language || "").trim();
            if (!durationSec && d?.duration) durationSec = Number(d.duration) || 0;
            enginesUsed.push(wModel === "whisper-large-v3" ? "Groq Whisper v3" : "Groq Whisper Turbo");
            break;
          }
        } else {
          errors.push(`Groq(${wModel}): HTTP ${r.status}`);
        }
      } catch (e) {
        errors.push(`Groq(${wModel}): ${e?.message || "timeout"}`);
      }
    }
  }

  // Convert to base64 only when safely <= 6.5MB to stay well inside Cloudflare Worker 128MB RAM
  const canBase64 = audioBuf.byteLength <= 6500000;
  const audioB64 = canBase64 ? arrayBufferToBase64(audioBuf) : "";

  // Engine 2: Cloudflare Workers AI @cf/openai/whisper-large-v3-turbo (fallback if Groq absent/failed)
  if (!whisperText && env.AI && canBase64) {
    try {
      const cfOut = await env.AI.run("@cf/openai/whisper-large-v3-turbo", {
        audio: audioB64,
        vad_filter: true,
      });
      const txt = String(cfOut?.text || "").trim();
      if (txt) {
        whisperText = txt;
        detectedLang = String(cfOut?.language || cfOut?.transcription_info?.language || "").trim();
        enginesUsed.push("CF Whisper v3 Turbo");
      }
    } catch (e) {
      errors.push(`CF-Whisper: ${e?.message || "err"}`);
    }
  }

  // Engine 3: Gemini Multimodal Native Audio — either cross-verifies Whisper draft directly against audio OR transcribes solo
  let finalTranscript = whisperText;
  if (canBase64 && getGeminiKeys(env).length) {
    const audioPrompt = whisperText
      ? "Listen carefully to this audio recording and inspect the automatic Whisper draft below.\n" +
        "Produce the 100% accurate, verbatim transcript in the exact spoken language(s) (keep mixed Persian + English technical words like CRISPR, Python, RNA-seq as spoken).\n" +
        "Fix any misheard words, Persian half-spaces (نیم‌فاصله), punctuation, and paragraph breaks.\n" +
        "STRICT RULES: Do NOT translate, do NOT summarize, do NOT add commentary or intro labels. Output ONLY the clean verbatim transcript text.\n\n" +
        `Whisper draft:\n"""\n${whisperText}\n"""`
      : "Listen to this audio recording and transcribe EVERY word verbatim in the exact spoken language(s) (keep mixed Persian + English terms as spoken).\n" +
        "Use accurate Persian half-spaces (نیم‌فاصله) and clean punctuation.\n" +
        "STRICT RULES: Do NOT translate, do NOT summarize, do NOT add any intro/outro text. Output ONLY the clean verbatim transcript.";

    const gemOut = await callGeminiMultiKey(
      env,
      ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
      [{ role: "user", parts: [{ inline_data: { mime_type: mimeType, data: audioB64 } }, { text: audioPrompt }] }],
      null,
      { temperature: 0.1, maxOutputTokens: 4096 },
      22000
    );
    const gTxt = String(gemOut?.text || "").trim();
    if (gTxt && gTxt.length >= Math.max(4, Math.floor(whisperText.length * 0.45))) {
      finalTranscript = gTxt;
      enginesUsed.push("Gemini Audio");
    } else if (gemOut?.errors?.length) {
      errors.push(gemOut.errors[0]);
    }
  } else if (whisperText) {
    // For huge files >6.5MB where Whisper already succeeded, do a fast text-only punctuation & ZWNJ polish
    const polishRes = await callGeminiMultiKey(
      env,
      ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash"],
      [{
        role: "user",
        parts: [{
          text:
            "Clean up ONLY punctuation, paragraph breaks, and Persian half-spaces (نیم‌فاصله) in this verbatim audio transcript without changing, adding, or translating any words. Output ONLY the cleaned text:\n\n" +
            whisperText,
        }],
      }],
      null,
      { temperature: 0.05, maxOutputTokens: 4096 },
      14000
    );
    if (polishRes?.text && polishRes.text.trim().length >= Math.floor(whisperText.length * 0.6)) {
      finalTranscript = polishRes.text.trim();
      enginesUsed.push("Gemini Polish");
    }
  }

  if (!finalTranscript) {
    return {
      ok: false,
      error: `تمام موتورهای تبدیل صدا ناموفق بودند (${errors.slice(0, 3).join(" | ") || "فایل خیلی بزرگ یا بی‌صدا است"})`,
    };
  }

  return {
    ok: true,
    transcript: finalTranscript,
    language: detectedLang || "تشخیص خودکار",
    durationSec,
    enginesUsed,
  };
}

async function callGeminiMultiKey(env, models, contents, systemInstruction, generationConfig = {}, timeoutMs = 10000) {
  const keys = getGeminiKeys(env);
  if (!keys.length) return { text: null, errors: ["No GEMINI_API_KEY configured"] };

  const startOffset = Math.floor(Date.now() / 1000) % keys.length;
  const orderedKeys = keys.map((_, i) => keys[(startOffset + i) % keys.length]);
  const errors = [];
  let totalFetches = 0;

  for (const model of models) {
    let model503Count = 0;
    const isLite = model.includes("lite");
    const keysForThisModel = isLite ? orderedKeys : orderedKeys.slice(0, Math.min(3, orderedKeys.length));
    for (let kIdx = 0; kIdx < keysForThisModel.length; kIdx++) {
      if (totalFetches >= 18) {
        return { text: null, errors };
      }
      const key = keysForThisModel[kIdx];
      const keyTag = `key#${((startOffset + kIdx) % keys.length) + 1}`;
      totalFetches++;
      try {
        const cfg = { ...generationConfig };
        if (model.includes("3.8") || model.includes("3.6")) {
          cfg.thinkingConfig = { thinkingLevel: "low" };
        }
        const body = { contents, generationConfig: cfg };
        if (systemInstruction) {
          body.system_instruction = { parts: [{ text: systemInstruction }] };
        }
        let resp = await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            signal: AbortSignal.timeout(timeoutMs),
            body: JSON.stringify(body),
          }
        );
        if (resp.status === 400 && cfg.thinkingConfig) {
          delete cfg.thinkingConfig;
          body.generationConfig = cfg;
          totalFetches++;
          resp = await fetch(
            `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              signal: AbortSignal.timeout(timeoutMs),
              body: JSON.stringify(body),
            }
          );
        }
        if (resp.status === 404) {
          const errT = await resp.text().catch(() => "");
          errors.push(`${model}: 404 ${errT.replace(/\s+/g, " ").slice(0, 70)}`);
          break;
        }
        if (resp.status === 503) {
          model503Count++;
          const errT = await resp.text().catch(() => "");
          if (model503Count === 1) {
            errors.push(`${model} ${keyTag}: 503 ${errT.replace(/\s+/g, " ").slice(0, 75)}`);
          }
          // 503 is global model overload — don't burn remaining keys on this model
          if (!isLite || model503Count >= 2) break;
          continue;
        }
        if (!resp.ok) {
          const errT = await resp.text().catch(() => "");
          errors.push(`${model} ${keyTag}: ${resp.status} ${errT.replace(/\s+/g, " ").slice(0, 85)}`);
          continue;
        }
        const data = await resp.json();
        const out = data?.candidates?.[0]?.content?.parts?.[0]?.text;
        if (out && out.trim()) return { text: out.trim(), errors };
        errors.push(`${model} ${keyTag}: empty candidates`);
      } catch (e) {
        errors.push(`${model} ${keyTag}: ${e?.message || "fetch error"}`);
      }
    }
  }
  return { text: null, errors };
}

async function parseNewsRequest(text, env) {
  const norm = toEnglishDigits(text).trim();
  const numMatch = norm.match(/\b(10|[1-9])\b/);
  const fallbackCount = numMatch ? Math.min(10, Math.max(1, parseInt(numMatch[1], 10))) : 5;
  const fillerRe = /(?:^|\s)(لطفا|لطفاً|سلام|ربات|بات|امروز|یه|یک|تعداد|تا|عدد|خبرهای|خبرها|خبر|اخبار|مقاله|مقالات|جدیدترین|جدید|تازه‌ترین|تازه|بفرست|بده|بگو|پیدا کن|در مورد|درباره|راجع به|مربوط به|از|رو|را|های|میخوام|می‌خوام|برام)(?=\s|$|[؟?!،.])/g;
  const cleanedTopic = ` ${norm} `
    .replace(/\b(10|[1-9])\b/g, " ")
    .replace(fillerRe, " ")
    .replace(fillerRe, " ")
    .replace(/[؟?!،.]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  const gemRes = await callGeminiMultiKey(
    env,
    ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash"],
    [
      {
        role: "user",
        parts: [
          {
            text:
              `Extract the EXACT news search topic and count from: "${text}".\n` +
              `Rules: count 1..10 (default 5). english_topic must be exact/specific (NEVER broaden "پاروویروس" to "virus" or "سندرم براون اسکوارد" to "neurology"). Return JSON.`,
          },
        ],
      },
    ],
    null,
    {
      temperature: 0.1,
      responseMimeType: "application/json",
      responseSchema: {
        type: "OBJECT",
        properties: {
          count: { type: "INTEGER" },
          english_topic: { type: "STRING" },
          persian_topic: { type: "STRING" },
        },
        required: ["count", "english_topic", "persian_topic"],
      },
    }
  );
  if (gemRes?.text) {
    try {
      const p = JSON.parse(gemRes.text);
      p.count = Math.min(10, Math.max(1, Number(p.count) || fallbackCount));
      return p;
    } catch (_) {}
  }
  return {
    count: fallbackCount,
    english_topic: cleanedTopic || "bioinformatics biology AI",
    persian_topic: cleanedTopic || "حوزه‌های منتخب",
  };
}

async function runConversationalCollab({ env, userText, history, repliedMessage, matchedRepos, matchedCourseContext, knowledgeModeTag }) {
  const systemPrompt = `تو همکار و مشاور محتواسازی «آکادمی بیوانفورماتیک ایران» (به مدیریت معین شیرین‌کام) هستی.
کاربر مثل یک همکار واقعی با تو گپ می‌زند تا روی ایده، متن آموزشی، اصلاح پست، یا جمع‌بندی نهایی به نتیجه برسد.

اصول قطعی رفتار و لحن تو:
1. **مکالمهٔ طبیعی و منعطف (ضد قالب تکراری):** همیشه یک فرم تکراری تحویل نده! دقیقاً به همان چیزی جواب بده که کاربر الان از تو می‌خواهد (اگر سؤال پرسید جواب بده، اگر پیش‌نویس خواست بده، اگر گفت بند دوم را عوض کن فقط همان را بهتر کن، اگر گفت نهاییش کن نسخه نهایی را آماده کن).
2. **سبک نوشتار وقتی متن آموزشی/پست می‌نویسی:**
   - **خیلی کوتاه، کوبنده، موردی (۱، ۲، ۳...)، واضح، غیرمبهم و کاملاً مفهوم.**
   - **کمتر تخصصی و قابل فهم:** اصطلاحات سنگین دانشگاهی یا کلمات انگلیسیِ توضیح‌داده‌نشده نیاور؛ اصل مفهوم را مستقیم و شفاف بگو (بدون داستان‌سرایی دور از موضوع و بدون مقدمه‌چینی اضافی).
   - **ضد زرد و ضد کلیشه:** جملات زرد مثل «معجزه»، «باورنکردنی»، «انفجار علم» اکیداً ممنوع. لحن علمیِ تمیز، باوقار و خودمونی باشد.
   - **ایموجی به‌جا:** از ایموجی‌های تمیز (🧬 🔹 💡 ⚡️ 🎯) برای خوانایی استفاده کن که متن خشک نباشد.
3. **تلگرام vs اینستاگرام:**
   - پیش‌فرض تلگرامی است (کوتاه و شفاف، ولی دقیق).
   - اگر کاربر گفت «اینستاگرامی» یا «اسلاید»: خطوط بسیار کوتاه و + یک کپشن خیلی کوتاه (نهایتاً ۳–۴ خط + هشتگ).
4. **انتخاب هوشمند منبع و اطاعت دقیق از دستور کاربر:**
   - **قانون طلایی:** هرگز خودسرانه فقط به یک منبع قفل نشو؛ دقیقاً مطابق دستور کاربر عمل کن:
     • اگر فقط [اطلاعات رسمی دوره‌ها...] داده شده و کاربر به دوره خودتان اشاره کرده: دقیقاً طبق سرفصل‌ها، ساعات، ابزارها و اساتید واقعی آکادمی بنویس (بدون جعل اسم یا عدد).
     • اگر کاربر به «بقیه منابع / منابع بیرونی / گیت‌هاب / منابع علمی دیگه» اشاره کرده (یا اصلاً اشاره‌ای به دوره نکرده): از دانش علمی خودت و [منابع واقعی کتابخانه گیت‌هاب] کمک بگیر و پای دوره را بی‌دلیل وسط نکش.
     • اگر کاربر خواسته «ترکیبی از هر دو» باشد: سرفصل/جزئیات واقعی دوره خودتان را هوشمندانه با ابزارها و منابع تکمیلی گیت‌هاب/علمی ترکیب کن.
     • **هر قید، فرمت، لحن یا خواسته‌ای که کاربر الان مشخص کرده بر همه چیز اولویت مطلق دارد.**
5. **امضا و منابع:**
   - امضای آکادمی (⭕️ معین شیرین کام / آکادمی بیوانفورماتیک ایران 🖥⭐️ / @ir_bioinformtic_academy) را **فقط وقتی کاربر گفت نهایی کن / با امضا / آماده انتشار** اضافه کن، نه وسط چت عادی!
   - هرگز اسم ریپوی گیت‌هاب از خودت نساز؛ فقط اگر در کانتکست منابع واقعی داده شده از همان‌ها استفاده کن.
   - اگر کاربر درخواست تولید تصویر غیرخبری (مثل پوستر، اسلاید گرافیکی یا تصویرسازی آموزشی) داد، شفاف و کوتاه بگو که طراحی گرافیکی و تصویرسازی در این ربات فقط برای **بخش اخبار علمی** فعال است و برای محتوای آموزشی بهترین متن، اسلاید متنی و کپشن را آماده می‌کنی.
6. **قالب HTML تلگرام:** فقط تگ‌های <b>، <i>، <u>، <blockquote>، <code> و <a href="..."> را به کار ببر (هرگز مارک‌داون ** یا ### یا تگ <br> ننویس).`;

  const contents = [];
  for (const turn of history || []) {
    contents.push({
      role: turn.role === "model" ? "model" : "user",
      parts: [{ text: turn.text }],
    });
  }

  let currentTurn = userText;
  if (repliedMessage) {
    currentTurn = `پیامی که کاربر روی آن ریپلای زده:\n"""\n${repliedMessage}\n"""\n\nپیام/درخواست فعلی کاربر روی این متن:\n${userText}`;
  }
  if (knowledgeModeTag) {
    currentTurn += `\n\n[${knowledgeModeTag}]`;
  }
  if (matchedCourseContext) {
    currentTurn += `\n\n[اطلاعات رسمی دوره‌ها، سرفصل‌ها و اساتید آکادمی — کاملاً دقیق و استخراج‌شده از کاتالوگ رسمی:\n${matchedCourseContext}\n]`;
  }
  if (matchedRepos) {
    currentTurn += `\n\n[منابع واقعی استخراج‌شده از کتابخانه ۱۵۰ ریپوی آکادمی — در صورت نیاز فقط از همین لینک‌های معتبر بده:\n${JSON.stringify(matchedRepos)}]`;
  }

  contents.push({ role: "user", parts: [{ text: currentTurn }] });

  // 1) Try Gemini 3.8 Flash (low thinking) -> 3.6 -> 3.5-lite -> 3.1-lite -> 3-flash-preview -> 2.5-flash-lite
  const gemRes = await callGeminiMultiKey(
    env,
    [
      "gemini-3.8-flash",
      "gemini-3.6-flash",
      "gemini-3.5-flash-lite",
      "gemini-3.1-flash-lite",
      "gemini-3-flash-preview",
      "gemini-2.5-flash-lite",
      "gemini-2.5-flash",
    ],
    contents,
    systemPrompt,
    { temperature: 0.45, maxOutputTokens: 2048 }
  );
  if (gemRes?.text) return cleanTelegramHtml(gemRes.text);

  // 2) Failover: Cloudflare Workers AI binding -> REST CF AI -> Groq -> OpenRouter
  const oaiMessages = [
    { role: "system", content: systemPrompt },
    ...(history || []).map((t) => ({ role: t.role === "model" ? "assistant" : "user", content: t.text })),
    { role: "user", content: currentTurn },
  ];

  const extraErrors = [];
  if (env.AI) {
    for (const cfModel of ["@cf/meta/llama-3.3-70b-instruct-fp8-fast", "@cf/google/gemma-3-12b-it"]) {
      try {
        const aiRes = await env.AI.run(cfModel, {
          messages: oaiMessages,
          max_tokens: 2048,
          temperature: 0.4,
        });
        const txt = (aiRes?.response || aiRes?.choices?.[0]?.message?.content || "").trim();
        if (txt) return cleanTelegramHtml(txt);
      } catch (e) {
        extraErrors.push(`env.AI(${cfModel}): ${e?.message || "err"}`);
      }
    }
  }

  const providers = [];
  if (env.CF_ACCOUNT_ID && env.CF_AI_TOKEN) {
    providers.push({
      ep: `https://api.cloudflare.com/client/v4/accounts/${env.CF_ACCOUNT_ID}/ai/v1/chat/completions`,
      key: env.CF_AI_TOKEN,
      model: "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    });
  }
  if (env.GROQ_API_KEY) {
    providers.push({
      ep: "https://api.groq.com/openai/v1/chat/completions",
      key: env.GROQ_API_KEY,
      model: "llama-3.3-70b-versatile",
    });
  }
  if (env.OPENROUTER_API_KEY) {
    providers.push({
      ep: "https://openrouter.ai/api/v1/chat/completions",
      key: env.OPENROUTER_API_KEY,
      model: "openrouter/free",
    });
  }

  for (const p of providers) {
    try {
      const r = await fetch(p.ep, {
        method: "POST",
        headers: { Authorization: `Bearer ${p.key}`, "Content-Type": "application/json" },
        signal: AbortSignal.timeout(15000),
        body: JSON.stringify({ model: p.model, temperature: 0.4, max_tokens: 2048, messages: oaiMessages }),
      });
      if (!r.ok) {
        extraErrors.push(`${p.model}: ${r.status}`);
        continue;
      }
      const d = await r.json();
      const txt = d?.choices?.[0]?.message?.content?.trim();
      if (txt) return cleanTelegramHtml(txt);
    } catch (e) {
      extraErrors.push(`${p.model}: ${e?.message || "err"}`);
    }
  }

  const allErrs = [...(gemRes?.errors || []), ...extraErrors];
  const diag = escapeHtml(allErrs.slice(0, 8).join("\n"));
  return `⚠️ خطا در اتصال به کلیدهای هوش مصنوعی.\n\n<code>${diag || "No error detail"}</code>`;
}

async function loadChatMemory(env, chatId) {
  if (!env.CHAT_MEMORY) return { history: [], last_draft: "" };
  try {
    const raw = await env.CHAT_MEMORY.get(`chat:${chatId}`);
    if (raw) {
      const parsed = JSON.parse(raw);
      return {
        history: Array.isArray(parsed.history) ? parsed.history : [],
        last_draft: String(parsed.last_draft || ""),
      };
    }
  } catch (_) {}
  return { history: [], last_draft: "" };
}

async function saveChatMemory(env, chatId, history, lastDraft) {
  if (!env.CHAT_MEMORY) return;
  try {
    await env.CHAT_MEMORY.put(
      `chat:${chatId}`,
      JSON.stringify({
        history: history.slice(-14),
        last_draft: lastDraft || "",
      }),
      { expirationTtl: 604800 }
    );
  } catch (_) {}
}

async function clearChatMemory(env, chatId) {
  if (!env.CHAT_MEMORY) return;
  try {
    await env.CHAT_MEMORY.delete(`chat:${chatId}`);
  } catch (_) {}
}

async function fetchBioinfoLibrary(env) {
  if (env.CHAT_MEMORY) {
    try {
      const cached = await env.CHAT_MEMORY.get("lib:bioinfo_v1");
      if (cached) return JSON.parse(cached);
    } catch (_) {}
  }
  const repo = String(env.GITHUB_REPO || "").trim();
  const token = String(env.GITHUB_TOKEN || "").trim();
  if (!repo) return null;

  try {
    const headers = {
      Accept: "application/vnd.github.raw",
      "User-Agent": "IranBioinfoWorker/1.0",
    };
    if (token) headers.Authorization = `Bearer ${token}`;
    const r = await fetch(`https://api.github.com/repos/${repo}/contents/knowledge/bioinfo_library.json`, {
      headers,
      signal: AbortSignal.timeout(5000),
    });
    if (!r.ok) return null;
    const lib = await r.json();
    if (env.CHAT_MEMORY) {
      try {
        await env.CHAT_MEMORY.put("lib:bioinfo_v1", JSON.stringify(lib), { expirationTtl: 21600 });
      } catch (_) {}
    }
    return lib;
  } catch (_) {
    return null;
  }
}

function findMatchingRepos(lib, queryText) {
  const tokens = String(queryText || "")
    .toLowerCase()
    .split(/[\s,،.؛؟!:/]+/)
    .filter((w) => w.length >= 3);
  const scored = [];
  for (const r of lib.repos || []) {
    const blob = `${r.cat || ""} ${r.repo || ""} ${r.desc || ""} ${r.lang || ""}`.toLowerCase();
    let sc = 0;
    for (const t of tokens) {
      if (blob.includes(t)) sc += (r.repo || "").toLowerCase().includes(t) ? 5 : 3;
    }
    if (sc > 0) scored.push({ sc, r });
  }
  scored.sort((a, b) => b.sc - a.sc);
  const topRepos = scored.slice(0, 6).map((x) => ({
    repo: x.r.repo,
    url: `https://github.com/${x.r.repo}`,
    desc: x.r.desc,
    cat: x.r.cat,
  }));
  return {
    repos: topRepos.length ? topRepos : (lib.repos || []).slice(0, 5).map((r) => ({ repo: r.repo, url: `https://github.com/${r.repo}`, desc: r.desc })),
    topics: (lib.github_topics || []).slice(0, 4),
    iran_searches: (lib.iran_persian_searches || []).slice(0, 3),
  };
}

async function fetchAcademyCourses(env) {
  let kb = null;
  if (env.CHAT_MEMORY) {
    try {
      const cached = await env.CHAT_MEMORY.get("kb:academy_courses_v1");
      if (cached) kb = JSON.parse(cached);
    } catch (_) {}
  }
  if (!kb) {
    const repo = String(env.GITHUB_REPO || "").trim();
    const token = String(env.GITHUB_TOKEN || "").trim();
    if (repo) {
      try {
        const headers = {
          Accept: "application/vnd.github.raw",
          "User-Agent": "IranBioinfoWorker/1.0",
        };
        if (token) headers.Authorization = `Bearer ${token}`;
        const r = await fetch(`https://api.github.com/repos/${repo}/contents/knowledge/academy_courses.json`, {
          headers,
          signal: AbortSignal.timeout(5000),
        });
        if (r.ok) {
          kb = await r.json();
          if (env.CHAT_MEMORY) {
            try {
              await env.CHAT_MEMORY.put("kb:academy_courses_v1", JSON.stringify(kb), { expirationTtl: 21600 });
            } catch (_) {}
          }
        }
      } catch (_) {}
    }
  }
  if (env.CHAT_MEMORY) {
    try {
      const customRaw = await env.CHAT_MEMORY.get("kb:academy_courses_custom");
      if (customRaw) {
        const customNotes = JSON.parse(customRaw);
        if (Array.isArray(customNotes) && customNotes.length) {
          kb = kb ? { ...kb, custom_notes: customNotes } : { academy: { name: "آکادمی بیوانفورماتیک ایران" }, courses: [], custom_notes: customNotes };
        }
      }
    } catch (_) {}
  }
  return kb;
}

async function appendCustomCourseNote(env, noteText) {
  if (!env.CHAT_MEMORY) return 0;
  let notes = [];
  try {
    const raw = await env.CHAT_MEMORY.get("kb:academy_courses_custom");
    if (raw) notes = JSON.parse(raw);
    if (!Array.isArray(notes)) notes = [];
  } catch (_) {
    notes = [];
  }
  notes.push({
    added_at: new Date().toISOString().slice(0, 10),
    text: String(noteText || "").trim().slice(0, 2000),
  });
  notes = notes.slice(-25);
  await env.CHAT_MEMORY.put("kb:academy_courses_custom", JSON.stringify(notes));
  return notes.length;
}

function formatAcademyOverviewHtml(kb) {
  if (!kb) {
    return "⚠️ فایل دانش دوره‌های آکادمی هنوز روی گیت‌هاب همگام‌سازی نشده است (مطمئن شوید فایل <code>knowledge/academy_courses.json</code> پوش شده باشد).";
  }
  const ac = kb.academy || {};
  const courses = kb.courses || [];
  const step1 = courses.find((c) => c.id === "step_1_intro");
  const step2 = courses.find((c) => c.id === "step_2_omics_ai");
  let out = `🎓 <b>پایگاه دانش رسمی ${escapeHtml(ac.name || "آکادمی بیوانفورماتیک ایران")}</b>\n`;
  if (ac.scientific_supervisor) {
    out += `👨‍🏫 <b>ناظر علمی:</b> ${escapeHtml(ac.scientific_supervisor.split("(")[0].trim())}\n`;
  }
  if (ac.registration_url) {
    out += `🌐 <b>ثبت‌نام:</b> ${escapeHtml(ac.registration_url)}\n`;
  }
  if (step1) {
    out += `\n📘 <b>${escapeHtml(step1.title)}</b> (${escapeHtml(step1.format || "")})\n`;
    out += `• ${(step1.topics || []).slice(0, 8).map(escapeHtml).join("، ")} و ...\n`;
  }
  if (step2) {
    const mods = step2.modules || [];
    const totalHours = mods.reduce((acc, m) => acc + (Number(m.hours) || 0), 0);
    out += `\n🚀 <b>${escapeHtml(step2.title)}</b>\n`;
    out += `📊 ${toPersianDigits(mods.length)} بخش تخصصی (~${toPersianDigits(totalHours)} ساعت آموزش):\n`;
    out += mods
      .map((m, idx) => {
        const inst = (m.instructors || []).map((i) => i.name).join(" و ");
        return `${toPersianDigits(idx + 1)}. <b>${escapeHtml(m.title)}</b> (${toPersianDigits(m.hours || 0)} ساعت)${inst ? ` — ${escapeHtml(inst)}` : ""}`;
      })
      .join("\n");
  }
  if (Array.isArray(kb.custom_notes) && kb.custom_notes.length) {
    out += `\n\n📌 <b>یادداشت‌های تکمیلی ثبت‌شده:</b> ${toPersianDigits(kb.custom_notes.length)} مورد`;
  }
  out += `\n\n💡 برای افزودن دوره جدید یا آپدیت تخفیف/زمان‌بندی: <code>/setcourse متن...</code>`;
  return out.slice(0, 3950);
}

function findMatchingCourseContext(kb, queryText) {
  if (!kb) return null;
  const q = String(queryText || "").toLowerCase();
  if (!q.trim()) return null;

  const courses = kb.courses || [];
  const step1 = courses.find((c) => c.id === "step_1_intro");
  const step2 = courses.find((c) => c.id === "step_2_omics_ai");
  const mods = step2?.modules || [];

  const stopWords = new Set([
    "دکتر", "مهندس", "استاد", "اساتید", "مدرس", "دوره", "بخش", "جلسه", "ساعت", "سرفصل", "سرفصلها",
    "درس", "کدوم", "کدام", "میده", "میدهد", "چیا", "چیست", "هست", "داره", "داری", "کن", "بساز",
    "بنویس", "معرفی", "پست", "استوری", "اسلاید", "کارگاه", "طرح", "ملی", "برای", "درباره", "روی",
    "این", "های", "ها", "که", "با", "از", "به", "را", "رو", "در", "تا", "بر", "یا", "یک", "یه", "ما", "خودمون", "هم",
    "دانشگاه", "تهران", "ایران", "بیوانفورماتیک", "دکتری", "دانشیار", "استادیار", "علوم", "پزشکی"
  ]);
  const tokens = q
    .replace(/ي/g, "ی")
    .replace(/ك/g, "ک")
    .split(/[\s,،.؛؟!:/()«»‌]+/)
    .filter((w) => (w.length >= 3 || /^[a-z0-9]{1,2}$/.test(w)) && !stopWords.has(w));

  const scored = [];
  let instructorMatched = false;
  for (const m of mods) {
    const instText = (m.instructors || []).map((i) => `${i.name} ${i.bio}`).join(" ");
    const normInst = instText.toLowerCase().replace(/ي/g, "ی").replace(/ك/g, "ک");
    const normTitle = (m.title || "").toLowerCase().replace(/ي/g, "ی").replace(/ك/g, "ک");
    const blob = `${m.title} ${instText} ${(m.syllabus || []).join(" ")} ${(m.tools || []).join(" ")}`
      .toLowerCase()
      .replace(/ي/g, "ی")
      .replace(/ك/g, "ک");
    let sc = 0;
    for (const t of tokens) {
      if (t.length >= 3 && normInst.includes(t)) {
        instructorMatched = true;
      }
      if (blob.includes(t)) {
        if (normInst.includes(t)) sc += 8;
        else if (normTitle.includes(t)) sc += 6;
        else sc += 2;
      }
    }
    if (sc >= 6) scored.push({ sc, m });
  }
  scored.sort((a, b) => b.sc - a.sc);

  const generalCourseIntent = /(دوره|طرح\s*ملی|سرفصل|سرفصل[‌ ]?ها|اساتید|استاد|مدرس|کارگاه|گام\s*اول|گام\s*دوم|مقدمات|فوق\s*جامع|ثبت[‌ ]?نام|اقساط|شهریه|گواهی|کارآموزی|آکادمی|خودمون|کتابچه)/i.test(q);
  if (!generalCourseIntent && !instructorMatched) {
    return null;
  }

  const lines = [];
  const ac = kb.academy || {};
  lines.push(`آکادمی: ${ac.name || "آکادمی بیوانفورماتیک ایران"} | ناظر علمی: ${ac.scientific_supervisor || ""} | ثبت‌نام: ${ac.registration_url || ""}`);

  if (scored.length > 0) {
    const maxSc = scored[0].sc;
    const minKeep = maxSc >= 12 ? Math.floor(maxSc * 0.6) : 6;
    const topMods = scored.filter((x) => x.sc >= minKeep).slice(0, 5).map((x) => x.m);
    lines.push("بخش‌های مرتبط در دوره فوق جامع (گام دوم):");
    for (const m of topMods) {
      const instStr = (m.instructors || []).map((i) => `${i.name} (${i.bio})`).join("؛ ");
      lines.push(
        `• ${m.title} (${m.sessions} جلسه / ${m.hours} ساعت)` +
        (instStr ? ` | مدرس: ${instStr}` : "") +
        (m.tools?.length ? ` | ابزارها: ${m.tools.join(", ")}` : "") +
        `\n  سرفصل: ${(m.syllabus || []).join(" | ")}`
      );
    }
  } else if (generalCourseIntent) {
    if (step1) {
      lines.push(`${step1.title} (${step1.format}): ویژگی‌ها: ${(step1.features || []).join("، ")} | محورها: ${(step1.topics || []).join("، ")}`);
    }
    if (step2) {
      const modSummary = mods
        .map((m) => {
          const inst = (m.instructors || []).map((i) => i.name).join(" و ");
          return `${m.title} (${m.hours}س${inst ? ` - ${inst}` : ""})`;
        })
        .join(" | ");
      lines.push(`${step2.title} — ${mods.length} بخش: ${modSummary}`);
    }
  }

  if (Array.isArray(kb.custom_notes) && kb.custom_notes.length) {
    lines.push("اطلاعات تکمیلی ثبت‌شده توسط ادمین:");
    for (const n of kb.custom_notes.slice(-5)) {
      lines.push(`- (${n.added_at}) ${n.text}`);
    }
  }

  return lines.join("\n").slice(0, 3200);
}

function resolveKnowledgeSources({ queryText, contextSeed = "", kb = null, lib = null }) {
  const q = String(queryText || "");
  const combinedSeed = `${q} ${contextSeed}`.trim();

  const excludeCourse = /(بدون\s*(?:اشاره\s*به\s*)?دوره|ربطی\s*به\s*دوره\s*ندار|از\s*دوره\s*(?:ما\s*)?(?:نگو|نیار|استفاده\s*نکن)|فقط\s*(?:منابع\s*دیگه|بقیه\s*منابع|منابع\s*بیرونی|گیت[‌ ]?هاب))/i.test(q);
  const wantsExternal = /(منابع|منبع|بقیه\s*منابع|سایر\s*منابع|منابع\s*دیگه|منابع\s*بیرونی|کتابخانه|ریپو|ریپوها|گیت[‌ ]?هاب|مخزن|لینک|ابزارهای\s*دیگه|تکمیلی)/i.test(q);
  const wantsHybrid = /(ترکیب|ترکیبی|هر\s*دو|هردو|هم\s+از\s+دوره.*هم|هم\s+سرفصل.*هم|در\s*کنار\s*(?:دوره|سرفصل)|به\s*اضافه\s*منابع)/i.test(q);

  const courseContext = (!excludeCourse && kb) ? findMatchingCourseContext(kb, combinedSeed) : null;
  const shouldFetchRepos = Boolean(lib) && (excludeCourse || wantsExternal || wantsHybrid);
  const matchedRepos = shouldFetchRepos ? findMatchingRepos(lib, combinedSeed) : null;

  let modeTag = "";
  if (excludeCourse) {
    modeTag = "حالت درخواستی کاربر: فقط از منابع علمی و کتابخانه گیت‌هاب استفاده کن و پای دوره آکادمی را وسط نکش.";
  } else if (wantsHybrid || (courseContext && matchedRepos)) {
    const finalCourseCtx = courseContext || (kb ? findMatchingCourseContext(kb, `دوره ${combinedSeed}`) : null);
    return {
      courseContext: finalCourseCtx,
      matchedRepos,
      modeTag: "حالت درخواستی کاربر: ترکیبی هوشمندانه از اطلاعات دوره آکادمی + سایر منابع علمی و گیت‌هاب — هر دو را مطابق دستور کاربر تلفیق کن.",
    };
  } else if (courseContext) {
    modeTag = "حالت درخواستی کاربر: منطبق بر دوره و سرفصل‌های خود آکادمی (دقیقاً از همین اطلاعات رسمی استفاده کن و دستور کاربر را مو به مو اجرا کن).";
  } else if (matchedRepos) {
    modeTag = "حالت درخواستی کاربر: استفاده از سایر منابع علمی و کتابخانه گیت‌هاب آکادمی.";
  }

  return { courseContext, matchedRepos, modeTag };
}

function cleanTelegramHtml(text) {
  let s = String(text || "")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/p\s*>/gi, "\n\n")
    .replace(/<p\s*>/gi, "")
    // Convert accidental markdown bold/headers into Telegram HTML
    .replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>")
    .replace(/^#{1,3}\s+(.+)$/gm, "<b>$1</b>")
    .replace(/<(?!\/?(?:b|strong|i|em|u|ins|s|strike|del|code|pre|blockquote|a)(?:\s|>))[^>]+>/gi, "");
  return s.trim();
}

async function triggerGitHubWorkflow(env, inputs) {
  const repo = (env.GITHUB_REPO || "").trim();
  const token = (env.GITHUB_TOKEN || "").trim();
  if (!repo || !token) {
    return { ok: false, error: "GITHUB_REPO or GITHUB_TOKEN is missing in Cloudflare." };
  }

  const headers = {
    Authorization: `Bearer ${token}`,
    Accept: "application/vnd.github+json",
    "User-Agent": "IranBioinfoWorker/1.0",
    "Content-Type": "application/json",
  };

  const listResp = await fetch(`https://api.github.com/repos/${repo}/actions/workflows`, { headers });
  if (!listResp.ok) {
    const errText = await listResp.text();
    return { ok: false, error: `GitHub list workflows (${listResp.status}): ${errText.slice(0, 180)}` };
  }
  const listData = await listResp.json();
  const workflows = listData.workflows || [];
  const activeWf = workflows.find((w) => w.state === "active") || workflows[0];
  if (!activeWf) {
    return { ok: false, error: "No GitHub Actions workflow found in repository." };
  }

  const repoResp = await fetch(`https://api.github.com/repos/${repo}`, { headers });
  const repoData = repoResp.ok ? await repoResp.json() : {};
  const defaultBranch = repoData.default_branch || "main";

  const dispatchResp = await fetch(
    `https://api.github.com/repos/${repo}/actions/workflows/${activeWf.id}/dispatches`,
    {
      method: "POST",
      headers,
      body: JSON.stringify({
        ref: defaultBranch,
        inputs,
      }),
    }
  );

  if (dispatchResp.status !== 204) {
    const errText = await dispatchResp.text();
    return { ok: false, error: `GitHub dispatch (${dispatchResp.status}): ${errText.slice(0, 180)}` };
  }

  return { ok: true };
}

async function checkGitHubActionsStatus(env) {
  const repo = (env.GITHUB_REPO || "").trim();
  const token = (env.GITHUB_TOKEN || "").trim();
  if (!repo || !token) {
    return "⚠️ متغیرهای <code>GITHUB_REPO</code> یا <code>GITHUB_TOKEN</code> در کلودفلر تنظیم نشده‌اند.";
  }
  const headers = {
    Authorization: `Bearer ${token}`,
    Accept: "application/vnd.github+json",
    "User-Agent": "IranBioinfoWorker/1.0",
  };
  try {
    const [repoResp, runsResp] = await Promise.all([
      fetch(`https://api.github.com/repos/${repo}`, { headers, signal: AbortSignal.timeout(6000) }),
      fetch(`https://api.github.com/repos/${repo}/actions/runs?per_page=4`, { headers, signal: AbortSignal.timeout(6000) }),
    ]);
    const repoData = repoResp.ok ? await repoResp.json() : null;
    if (!runsResp.ok) {
      const errTxt = await runsResp.text();
      return `⚠️ خطا در بررسی گیت‌هاب (${runsResp.status}):\n<code>${escapeHtml(errTxt.slice(0, 220))}</code>`;
    }
    const runsData = await runsResp.json();
    const runs = runsData.workflow_runs || [];
    const isPrivate = repoData?.private === true;
    const lines = [
      `🛠 <b>وضعیت زنده سرور گیت‌هاب (<code>${escapeHtml(repo)}</code>)</b>`,
      isPrivate
        ? "🔒 نوع مخزن: <b>Private</b> (سقف رایگان ماهانه گیت‌هاب برای مخزن خصوصی ۲٬۰۰۰ دقیقه است؛ اگر آخر ماه پیام‌ها روی ۲۰٪ گیر کنند، مخزن را Public کنید تا نامحدود شود!)"
        : "🌐 نوع مخزن: <b>Public</b> (اکشن نامحدود رایگان)",
      "",
    ];
    if (!runs.length) {
      lines.push("هیچ اجرای ثبت‌شده‌ای پیدا نشد.");
      return lines.join("\n");
    }
    for (let i = 0; i < Math.min(3, runs.length); i++) {
      const r = runs[i];
      const st = r.status || "?";
      const conc = r.conclusion || "در حال اجرا";
      const emoji = conc === "success" ? "✅" : (conc === "failure" || conc === "startup_failure" ? "❌" : "⏳");
      let detail = "";
      if (i === 0 && (conc === "failure" || conc === "startup_failure" || st === "queued")) {
        try {
          const jr = await fetch(`https://api.github.com/repos/${repo}/actions/runs/${r.id}/jobs`, {
            headers,
            signal: AbortSignal.timeout(5000),
          });
          if (jr.ok) {
            const jd = await jr.json();
            const failedJob = (jd.jobs || []).find((j) => j.conclusion === "failure");
            const failedStep = failedJob?.steps?.find((s) => s.conclusion === "failure");
            if (failedStep) detail = ` ← مرحلهٔ خطا: <code>${escapeHtml(failedStep.name)}</code>`;
            else if (st === "queued") detail = " ← در صف انتظار رانر گیت‌هاب (یا محدودیت دقایق مخزن Private)";
          }
        } catch (_) {}
      }
      lines.push(`${emoji} <b>#${r.run_number || r.id}</b> — <code>${escapeHtml(st)}</code> / <code>${escapeHtml(conc)}</code> (${escapeHtml(r.event || "")})${detail}`);
    }
    lines.push(`\n🔗 <a href="https://github.com/${escapeHtml(repo)}/actions">مشاهده لاگ کامل در GitHub Actions</a>`);
    return lines.join("\n");
  } catch (err) {
    return `⚠️ خطا در استعلام وضعیت گیت‌هاب: <code>${escapeHtml(err?.message || String(err))}</code>`;
  }
}

async function finalizeDispatchStatus(botToken, chatId, statusMsgId, reqDesc, triggered, gameMarkup) {
  if (!triggered.ok) {
    const errHtml = `⚠️ خطا در اتصال به گیت‌هاب:\n<code>${escapeHtml(triggered.error)}</code>`;
    if (statusMsgId) {
      await editTelegram(botToken, chatId, statusMsgId, errHtml);
    } else {
      await sendTelegram(botToken, chatId, errHtml);
    }
    return;
  }
  if (statusMsgId) {
    await editTelegram(
      botToken,
      chatId,
      statusMsgId,
      `${reqDesc}\n\n<code>[▓▓░░░░░░░░] ۲۰٪</code>\nسرور بیدار شد؛ در انتظار شروع پردازش...\n(اگر بیش از ۲ دقیقه اینجا ماند، دستور <code>/status</code> را بفرستید)\n\n💡 تا آماده شدن، بازی تقویت حافظه زیر را بزنید:`,
      gameMarkup
    );
  }
}

async function sendTyping(botToken, chatId) {
  try {
    await fetch(`https://api.telegram.org/bot${botToken}/sendChatAction`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chat_id: chatId, action: "typing" }),
    });
  } catch (_) {}
}

async function sendTelegram(botToken, chatId, htmlText, replyMarkup = null, replyToMsgId = null) {
  const full = String(htmlText || "");
  const chunks = [];
  for (let i = 0; i < full.length; i += 3900) {
    chunks.push(full.slice(i, i + 3900));
  }
  if (!chunks.length) chunks.push("...");

  let firstData = null;
  for (let idx = 0; idx < chunks.length; idx++) {
    const chunk = chunks[idx];
    const body = {
      chat_id: chatId,
      text: chunk,
      parse_mode: "HTML",
      disable_web_page_preview: true,
    };
    if (idx === 0 && replyMarkup) body.reply_markup = replyMarkup;
    if (idx === 0 && replyToMsgId) body.reply_to_message_id = replyToMsgId;

    let resp = await fetch(`https://api.telegram.org/bot${botToken}/sendMessage`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    let data = null;
    try { data = await resp.json(); } catch (_) {}

    if (!data || !data.ok) {
      // Retry without reply_markup/reply_to_message_id first, then fallback to plain text if HTML tags fail
      resp = await fetch(`https://api.telegram.org/bot${botToken}/sendMessage`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chat_id: chatId, text: chunk, parse_mode: "HTML", disable_web_page_preview: true }),
      });
      try { data = await resp.json(); } catch (_) {}
      if (!data || !data.ok) {
        const plain = chunk.replace(/<[^>]+>/g, "");
        resp = await fetch(`https://api.telegram.org/bot${botToken}/sendMessage`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ chat_id: chatId, text: plain, disable_web_page_preview: true }),
        });
        try { data = await resp.json(); } catch (_) {}
      }
    }
    if (idx === 0) firstData = data;
  }
  return firstData;
}

async function editTelegram(botToken, chatId, messageId, htmlText, replyMarkup = null) {
  const body = {
    chat_id: chatId,
    message_id: Number(messageId),
    text: htmlText,
    parse_mode: "HTML",
  };
  if (replyMarkup) body.reply_markup = replyMarkup;
  await fetch(`https://api.telegram.org/bot${botToken}/editMessageText`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function getBrainGameHtml() {
  return `<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <title>باشگاه هوش و حافظه آکادمی</title>
  <script src="https://telegram.org/js/telegram-web-app.js"></script>
  <style>
    :root {
      --bg: #080d1a; --card: #121a30; --card-2: #1a2544; --border: #2a3a66;
      --text: #eef2ff; --muted: #9fb0d3; --blue: #38bdf8; --green: #34d399;
      --amber: #fbbf24; --pink: #f472b6; --red: #fb7185;
    }
    * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
    body {
      margin: 0; padding: 14px 14px 36px; background: var(--bg); color: var(--text);
      font-family: Vazirmatn, Tahoma, Arial, sans-serif; user-select: none;
    }
    .wrap { max-width: 480px; margin: 0 auto; }
    .header {
      background: linear-gradient(135deg, #162244, #0f1730);
      border: 1px solid var(--border); border-radius: 16px; padding: 14px 16px; margin-bottom: 12px;
      display: flex; justify-content: space-between; align-items: center;
    }
    .header h1 { margin: 0; font-size: 16px; color: #fff; }
    .header small { color: var(--muted); font-size: 12px; display: block; margin-top: 2px; }
    .tabs {
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 12px;
    }
    .tab {
      background: var(--card); border: 1px solid var(--border); color: var(--muted);
      padding: 10px 6px; border-radius: 12px; font-family: inherit; font-size: 12.5px;
      font-weight: bold; cursor: pointer; text-align: center;
    }
    .tab.active {
      background: rgba(56,189,248,0.16); border-color: var(--blue); color: #fff;
    }
    .panel {
      display: none; background: var(--card); border: 1px solid var(--border);
      border-radius: 18px; padding: 16px;
    }
    .panel.active { display: block; }
    .stats {
      display: flex; justify-content: space-between; align-items: center;
      background: #0a1020; border: 1px solid var(--border); border-radius: 12px;
      padding: 8px 12px; margin-bottom: 14px; font-size: 13px;
    }
    .stats b { color: var(--green); }
    .status {
      text-align: center; min-height: 28px; font-size: 14px; font-weight: bold;
      color: var(--amber); margin-bottom: 12px;
    }
    .btn {
      width: 100%; background: linear-gradient(135deg, #0ea5e9, #38bdf8); color: #041224;
      border: none; border-radius: 12px; padding: 12px; font-family: inherit; font-size: 14.5px;
      font-weight: bold; cursor: pointer; margin-top: 10px;
    }
    .btn-green { background: linear-gradient(135deg, #10b981, #34d399); color: #042416; }

    /* Game 1: Memory Matrix */
    .matrix {
      display: grid; gap: 8px; max-width: 310px; margin: 0 auto 10px;
    }
    .cell {
      aspect-ratio: 1; border-radius: 12px; background: var(--card-2);
      border: 2px solid var(--border); cursor: pointer; transition: transform 0.12s, background 0.15s;
    }
    .cell:active { transform: scale(0.94); }
    .cell.lit { background: var(--blue); border-color: #bae6fd; box-shadow: 0 0 18px rgba(56,189,248,0.55); }
    .cell.good { background: var(--green); border-color: #a7f3d0; }
    .cell.bad { background: var(--red); border-color: #fecdd3; }

    /* Game 2: DNA Simon Sequence */
    .dna-grid {
      display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; max-width: 300px; margin: 0 auto 10px;
    }
    .dna-btn {
      aspect-ratio: 1.15; border-radius: 18px; border: 2px solid rgba(255,255,255,0.14);
      font-size: 28px; font-weight: 900; color: #fff; cursor: pointer;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      opacity: 0.72; transition: transform 0.1s, opacity 0.12s, box-shadow 0.12s;
    }
    .dna-btn span { font-size: 11px; font-weight: normal; opacity: 0.85; margin-top: 2px; }
    .dna-btn.flash { opacity: 1; transform: scale(1.04); }
    .dna-a { background: #0284c7; } .dna-a.flash { box-shadow: 0 0 24px #38bdf8; }
    .dna-t { background: #059669; } .dna-t.flash { box-shadow: 0 0 24px #34d399; }
    .dna-g { background: #d97706; } .dna-g.flash { box-shadow: 0 0 24px #fbbf24; }
    .dna-c { background: #db2777; } .dna-c.flash { box-shadow: 0 0 24px #f472b6; }

    /* Game 3: Stroop Color Challenge */
    .stroop-word {
      text-align: center; font-size: 40px; font-weight: 900; padding: 24px 10px;
      background: #0a1020; border: 1px solid var(--border); border-radius: 16px; margin-bottom: 14px;
    }
    .stroop-opts {
      display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;
    }
    .stroop-btn {
      background: var(--card-2); border: 1px solid var(--border); color: #fff;
      padding: 13px; border-radius: 12px; font-family: inherit; font-size: 15px; font-weight: bold; cursor: pointer;
    }
    .stroop-btn:active { transform: scale(0.96); }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <div>
        <h1>🧠 باشگاه هوش و حافظه</h1>
        <small>تا خبرها آماده شوند، مغزت را گرم کن!</small>
      </div>
      <div style="font-size:24px;">🧬</div>
    </div>

    <div class="tabs">
      <button class="tab active" onclick="switchTab('matrix', this)">🧩 ماتریس حافظه</button>
      <button class="tab" onclick="switchTab('dna', this)">🧬 دنباله DNA</button>
      <button class="tab" onclick="switchTab('stroop', this)">⚡ تست سرعت رنگ</button>
    </div>

    <!-- GAME 1: MEMORY MATRIX -->
    <div class="panel active" id="p-matrix">
      <div class="stats">
        <span>مرحله: <b id="m-lvl">۱</b></span>
        <span>بهترین رکورد: <b id="m-best">۰</b></span>
      </div>
      <div class="status" id="m-status">خانه‌های روشن را به خاطر بسپار!</div>
      <div class="matrix" id="m-grid"></div>
      <button class="btn" id="m-start" onclick="startMatrix()">شروع مرحله</button>
    </div>

    <!-- GAME 2: DNA SEQUENCE -->
    <div class="panel" id="p-dna">
      <div class="stats">
        <span>طول توالی: <b id="d-lvl">۰</b></span>
        <span>بهترین رکورد: <b id="d-best">۰</b></span>
      </div>
      <div class="status" id="d-status">توالی نوکلئوتیدها را تکرار کن</div>
      <div class="dna-grid">
        <button class="dna-btn dna-a" data-idx="0" onclick="tapDna(0)">A<span>آدنین</span></button>
        <button class="dna-btn dna-t" data-idx="1" onclick="tapDna(1)">T<span>تایمین</span></button>
        <button class="dna-btn dna-g" data-idx="2" onclick="tapDna(2)">G<span>گوانین</span></button>
        <button class="dna-btn dna-c" data-idx="3" onclick="tapDna(3)">C<span>سایتوزین</span></button>
      </div>
      <button class="btn btn-green" id="d-start" onclick="startDna()">شروع بازی DNA</button>
    </div>

    <!-- GAME 3: STROOP SPEED -->
    <div class="panel" id="p-stroop">
      <div class="stats">
        <span>امتیاز: <b id="s-score">۰</b></span>
        <span>زمان: <b id="s-time">۳۰</b> ثانیه</span>
        <span>رکورد: <b id="s-best">۰</b></span>
      </div>
      <div class="status" id="s-status">رنگِ واقعی نوشته را انتخاب کن (نه معنی کلمه را)!</div>
      <div class="stroop-word" id="s-word">آماده؟</div>
      <div class="stroop-opts" id="s-opts"></div>
      <button class="btn" id="s-start" onclick="startStroop()">شروع چالش ۳۰ ثانیه‌ای</button>
    </div>
  </div>

  <script>
    if (window.Telegram && window.Telegram.WebApp) {
      window.Telegram.WebApp.ready();
      window.Telegram.WebApp.expand();
    }
    const faDigits = (n) => String(n).replace(/\\d/g, d => "۰۱۲۳۴۵۶۷۸۹"[d]);
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    let actx = null;
    function beep(freq, dur = 0.12) {
      try {
        if (!actx && AudioCtx) actx = new AudioCtx();
        if (!actx) return;
        const o = actx.createOscillator();
        const g = actx.createGain();
        o.frequency.value = freq; o.type = "sine";
        g.gain.setValueAtTime(0.08, actx.currentTime);
        g.gain.exponentialRampToValueAtTime(0.001, actx.currentTime + dur);
        o.connect(g); g.connect(actx.destination); o.start(); o.stop(actx.currentTime + dur);
      } catch(e) {}
    }

    function switchTab(id, btn) {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('p-' + id).classList.add('active');
    }

    // ---- GAME 1: MEMORY MATRIX ----
    let mLvl = 1, mSize = 4, mTargets = new Set(), mPicked = new Set(), mCanClick = false;
    let mBest = Number(localStorage.getItem('bio_m_best') || 0);
    document.getElementById('m-best').textContent = faDigits(mBest);

    function renderMatrix() {
      mSize = mLvl >= 6 ? 5 : 4;
      const g = document.getElementById('m-grid');
      g.style.gridTemplateColumns = 'repeat(' + mSize + ', 1fr)';
      g.innerHTML = '';
      for (let i = 0; i < mSize * mSize; i++) {
        const c = document.createElement('div');
        c.className = 'cell';
        c.onclick = () => pickCell(i, c);
        g.appendChild(c);
      }
    }
    renderMatrix();

    function startMatrix() {
      mCanClick = false; mTargets.clear(); mPicked.clear();
      renderMatrix();
      document.getElementById('m-lvl').textContent = faDigits(mLvl);
      document.getElementById('m-status').textContent = 'خوب نگاه کن...';
      const count = Math.min(mSize * mSize - 3, mLvl + 2);
      while (mTargets.size < count) mTargets.add(Math.floor(Math.random() * mSize * mSize));
      const cells = document.getElementById('m-grid').children;
      mTargets.forEach(i => cells[i].classList.add('lit'));
      beep(520, 0.14);
      setTimeout(() => {
        mTargets.forEach(i => cells[i].classList.remove('lit'));
        mCanClick = true;
        document.getElementById('m-status').textContent = 'حالا خانه‌های روشن را پیدا کن! (' + faDigits(count) + ' خانه)';
      }, Math.max(950, 1750 - mLvl * 70));
    }

    function pickCell(i, el) {
      if (!mCanClick || mPicked.has(i)) return;
      if (mTargets.has(i)) {
        mPicked.add(i); el.classList.add('good'); beep(660 + mPicked.size * 40, 0.08);
        if (mPicked.size === mTargets.size) {
          mCanClick = false;
          if (mLvl > mBest) { mBest = mLvl; localStorage.setItem('bio_m_best', mBest); document.getElementById('m-best').textContent = faDigits(mBest); }
          document.getElementById('m-status').textContent = 'آفرین! مرحله بعد...';
          mLvl++;
          setTimeout(startMatrix, 800);
        }
      } else {
        mCanClick = false; el.classList.add('bad'); beep(200, 0.25);
        const cells = document.getElementById('m-grid').children;
        mTargets.forEach(t => { if (!mPicked.has(t)) cells[t].classList.add('lit'); });
        document.getElementById('m-status').textContent = 'اشتباه شد! رکورد این دور: مرحله ' + faDigits(mLvl);
        mLvl = 1;
      }
    }

    // ---- GAME 2: DNA SEQUENCE ----
    let dSeq = [], dUserIdx = 0, dLocked = true;
    let dBest = Number(localStorage.getItem('bio_d_best') || 0);
    document.getElementById('d-best').textContent = faDigits(dBest);
    const dnaFreqs = [440, 554, 659, 880];

    function startDna() {
      dSeq = []; dUserIdx = 0; dLocked = true;
      nextDnaRound();
    }
    function nextDnaRound() {
      dSeq.push(Math.floor(Math.random() * 4));
      dUserIdx = 0; dLocked = true;
      document.getElementById('d-lvl').textContent = faDigits(dSeq.length);
      document.getElementById('d-status').textContent = 'توالی را تماشا کن...';
      let i = 0;
      const timer = setInterval(() => {
        flashDna(dSeq[i]);
        i++;
        if (i >= dSeq.length) {
          clearInterval(timer);
          setTimeout(() => { dLocked = false; document.getElementById('d-status').textContent = 'نوبت توست!'; }, 380);
        }
      }, 620);
    }
    function flashDna(idx) {
      const btn = document.querySelector('.dna-btn[data-idx="' + idx + '"]');
      btn.classList.add('flash');
      beep(dnaFreqs[idx], 0.16);
      setTimeout(() => btn.classList.remove('flash'), 300);
    }
    function tapDna(idx) {
      if (dLocked) return;
      flashDna(idx);
      if (idx === dSeq[dUserIdx]) {
        dUserIdx++;
        if (dUserIdx === dSeq.length) {
          dLocked = true;
          if (dSeq.length > dBest) { dBest = dSeq.length; localStorage.setItem('bio_d_best', dBest); document.getElementById('d-best').textContent = faDigits(dBest); }
          document.getElementById('d-status').textContent = 'عالی بود! پایه بعدی اضافه شد...';
          setTimeout(nextDnaRound, 700);
        }
      } else {
        dLocked = true; beep(180, 0.3);
        document.getElementById('d-status').textContent = 'توالی اشتباه شد! امتیاز: ' + faDigits(Math.max(0, dSeq.length - 1));
      }
    }

    // ---- GAME 3: STROOP BRAIN SPEED ----
    const colors = [
      { name: 'آبی', hex: '#38bdf8' },
      { name: 'سبز', hex: '#34d399' },
      { name: 'زرد', hex: '#fbbf24' },
      { name: 'قرمز', hex: '#fb7185' }
    ];
    let sScore = 0, sTime = 30, sTimer = null, sAnswer = null, sActive = false;
    let sBest = Number(localStorage.getItem('bio_s_best') || 0);
    document.getElementById('s-best').textContent = faDigits(sBest);

    function renderStroopOpts() {
      const box = document.getElementById('s-opts');
      box.innerHTML = '';
      colors.forEach(c => {
        const b = document.createElement('button');
        b.className = 'stroop-btn';
        b.textContent = c.name;
        b.onclick = () => pickStroop(c.name);
        box.appendChild(b);
      });
    }
    renderStroopOpts();

    function nextStroop() {
      const textCol = colors[Math.floor(Math.random() * colors.length)];
      let inkCol = colors[Math.floor(Math.random() * colors.length)];
      if (Math.random() < 0.75) {
        while (inkCol.name === textCol.name) inkCol = colors[Math.floor(Math.random() * colors.length)];
      }
      const w = document.getElementById('s-word');
      w.textContent = textCol.name;
      w.style.color = inkCol.hex;
      sAnswer = inkCol.name;
    }

    function startStroop() {
      clearInterval(sTimer);
      sScore = 0; sTime = 30; sActive = true;
      document.getElementById('s-score').textContent = faDigits(0);
      document.getElementById('s-time').textContent = faDigits(30);
      document.getElementById('s-status').textContent = 'سریع رنگِ جوهر متن را بزن!';
      nextStroop();
      sTimer = setInterval(() => {
        sTime--;
        document.getElementById('s-time').textContent = faDigits(sTime);
        if (sTime <= 0) {
          clearInterval(sTimer); sActive = false;
          document.getElementById('s-word').textContent = 'پایان! امتیاز: ' + faDigits(sScore);
          document.getElementById('s-word').style.color = '#fff';
          if (sScore > sBest) { sBest = sScore; localStorage.setItem('bio_s_best', sBest); document.getElementById('s-best').textContent = faDigits(sBest); }
        }
      }, 1000);
    }

    function pickStroop(name) {
      if (!sActive) return;
      if (name === sAnswer) { sScore++; beep(720, 0.07); }
      else { sScore = Math.max(0, sScore - 1); beep(220, 0.12); }
      document.getElementById('s-score').textContent = faDigits(sScore);
      nextStroop();
    }
  </script>
</body>
</html>`;
}
