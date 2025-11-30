(() => {
  const questionEl = document.getElementById("question");
  const askButton = document.getElementById("ask-button");
  const statusEl = document.getElementById("status");
  const answerEl = document.getElementById("answer");
  const sourcesEl = document.getElementById("sources");

  const getApiBase = () => {
    if (window.location.origin && window.location.origin.startsWith("http")) {
      return window.location.origin;
    }
    return "http://localhost:8000";
  };

  const setStatus = (text, isError = false) => {
    statusEl.textContent = text || "";
    statusEl.classList.toggle("error", Boolean(isError));
  };

  const formatSourcePath = (path) => {
    if (!path) return "Unknown source";

    // Extract filename from path
    const parts = path.split("/");
    let filename = parts[parts.length - 1] || path;

    // Remove .md extension if present
    filename = filename.replace(/\.md$/, "");

    // Try to extract meaningful path segments (docs/tutorial/...)
    const docsIndex = path.indexOf("/docs/");
    if (docsIndex !== -1) {
      const relativePath = path.substring(docsIndex + 6);
      const pathParts = relativePath
        .split("/")
        .filter((p) => p && !p.endsWith(".md"));
      if (pathParts.length > 0) {
        return (
          pathParts.join(" › ") +
          (filename !== pathParts[pathParts.length - 1] ? ` › ${filename}` : "")
        );
      }
    }

    // Fallback: return just the filename
    return filename;
  };

  const renderAnswer = (text) => {
    answerEl.innerHTML = text
      .split("\n")
      .map((line) => {
        if (line.trim() === "") return "<br>";
        return `<p>${escapeHtml(line)}</p>`;
      })
      .join("");
  };

  const renderSources = (sources) => {
    sourcesEl.innerHTML = "";
    if (!sources || !sources.length) {
      sourcesEl.innerHTML = "<p class='empty-state'>No sources available.</p>";
      return;
    }

    sources.forEach((src) => {
      const sourceItem = document.createElement("div");
      sourceItem.className = "source-item";

      const sourcePath = document.createElement("div");
      sourcePath.className = "source-path";
      sourcePath.textContent = formatSourcePath(src.source);
      sourceItem.appendChild(sourcePath);

      // Add metadata if available
      if (src.category_path || src.top_level_category) {
        const sourceMeta = document.createElement("div");
        sourceMeta.className = "source-meta";
        const metaParts = [];
        if (src.category_path) metaParts.push(src.category_path);
        if (
          src.top_level_category &&
          !src.category_path?.includes(src.top_level_category)
        ) {
          metaParts.unshift(src.top_level_category);
        }
        sourceMeta.textContent = metaParts.join(" › ");
        sourceItem.appendChild(sourceMeta);
      }

      sourcesEl.appendChild(sourceItem);
    });
  };

  const escapeHtml = (text) => {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  };

  const ask = async () => {
    const question = questionEl.value.trim();
    if (!question) {
      setStatus("Please enter a question.", true);
      return;
    }

    askButton.disabled = true;
    setStatus("Thinking...");
    renderAnswer("");
    renderSources([]);

    try {
      const resp = await fetch(`${getApiBase()}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!resp.ok) {
        let detail = `Error ${resp.status}`;
        try {
          const data = await resp.json();
          if (data && data.detail) detail = data.detail;
        } catch {
          // ignore
        }
        throw new Error(detail);
      }

      const data = await resp.json();
      renderAnswer(data.answer || "");
      renderSources(data.sources || []);
      setStatus("");
    } catch (err) {
      console.error(err);
      setStatus(
        `Request failed: ${err && err.message ? err.message : "Unknown error"}`,
        true
      );
    } finally {
      askButton.disabled = false;
    }
  };

  askButton.addEventListener("click", ask);

  questionEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      if (!askButton.disabled) {
        ask();
      }
    }
  });

  // Handle premade question buttons
  document.querySelectorAll(".example-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const question = btn.getAttribute("data-question");
      if (question) {
        questionEl.value = question;
        questionEl.focus();
        setStatus("");
      }
    });
  });
})();
