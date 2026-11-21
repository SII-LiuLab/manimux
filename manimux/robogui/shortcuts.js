function installLetterShortcutToggle() {
  const tag = "manimux-letter-shortcuts";
  if (customElements.get(tag)) return;
  let enabled = true;
  try {
    enabled = sessionStorage.getItem(tag) !== "off";
  } catch { /* Storage is optional. */ }

  customElements.define(tag, class extends HTMLElement {
    connectedCallback() {
      this.innerHTML = '<label><input type="checkbox"> Enable letter shortcuts</label>';
      const checkbox = this.querySelector("input");
      checkbox.checked = enabled;
      checkbox.onchange = () => {
        enabled = checkbox.checked;
        try {
          sessionStorage.setItem(tag, enabled ? "on" : "off");
        } catch { /* Keep the choice in memory for this page. */ }
      };
    }
  });
}

function installShortcuts(buttonIds) {
  const installation = Symbol.for("manimux.robogui.shortcuts");
  // A client reconnect can reinstall these listeners while Enter is still down.
  const previewEnterKeys = window[installation]?.() ?? new Set();
  installLetterShortcutToggle();
  const listeners = new AbortController();
  const options = { signal: listeners.signal, capture: true };
  const checkedToggleSelector = "manimux-letter-shortcuts input:checked";
  // Viser shares button labels across viewers. Each tab renders its own key hints.
  const hintRule = (selector, key) =>
    `:root:has(${checkedToggleSelector}) ${selector}::after { content: " (${key})"; white-space: pre; }`;
  const hints = document.createElement("style");
  hints.textContent = [
    ...Object.entries(buttonIds).map(([key, id]) =>
      hintRule(`#${CSS.escape(id)} .mantine-Button-label`, key === "enter" ? "Enter" : key.toUpperCase())),
    hintRule(".manimux-reference-panel header strong", "T"),
    hintRule(".manimux-reference-preview button", "Esc"),
  ].join("\n");
  document.head.append(hints);

  const editor = [
    "input", "textarea", "select", "[contenteditable]:not([contenteditable=false])",
    "[role=combobox]", "[role=listbox]", "[role=slider]", "[role=spinbutton]", "[role=textbox]",
  ].join(",");
  const pageSurface = (node) =>
    node === document.body || node === document.documentElement || node instanceof HTMLCanvasElement;
  const popup = "dialog[open],[role=dialog],[role=alertdialog],[role=listbox],[role=menu]";
  const reference = "div:has(> .manimux-reference-anchor) + div img";
  let preview = null;

  const visible = (element) =>
    element?.getClientRects().length > 0 && getComputedStyle(element).visibility === "visible";
  const otherPopup = () =>
    Array.from(document.querySelectorAll(popup)).some((node) => node !== preview && visible(node));

  function openReference(source) {
    const dialog = document.createElement("dialog");
    dialog.className = "manimux-reference-preview";
    dialog.setAttribute("aria-label", "TOP · Layout reference");
    const close = document.createElement("button");
    close.type = "button";
    close.textContent = "Close";
    close.addEventListener("keydown", (event) => {
      // The native click closes the dialog and returns focus before the key repeats.
      if (event.isTrusted && event.key === "Enter" && !event.defaultPrevented) {
        previewEnterKeys.add(event.code);
      }
    });
    close.onclick = () => dialog.close();
    const image = document.createElement("img");
    image.alt = "TOP · Layout reference";
    image.src = source.src;
    dialog.append(close, image);
    document.body.append(dialog);
    preview = dialog;
    const observer = new MutationObserver(() => {
      if (!source.isConnected || otherPopup()) dialog.close();
      else if (image.src !== source.src) image.src = source.src;
    });
    observer.observe(source, { attributes: true, attributeFilter: ["src"] });
    observer.observe(document.body, { childList: true, subtree: true });
    dialog.addEventListener("close", () => {
      observer.disconnect();
      dialog.remove();
      if (preview === dialog) preview = null;
    }, { once: true });
    dialog.showModal();
  }

  document.addEventListener("click", (event) => {
    const button = event.target instanceof Element && event.target.closest("button");
    if (!button || preview || !document.querySelector(checkedToggleSelector)) return;
    const source = document.querySelector(reference);
    if (!source?.parentElement.contains(button) || otherPopup()) return;
    event.preventDefault();
    event.stopPropagation();
    openReference(source);
  }, options);

  document.addEventListener("keyup", (event) => {
    if (event.isTrusted && event.key === "Enter") previewEnterKeys.delete(event.code);
  }, options);

  document.addEventListener("keydown", (event) => {
    if (!event.isTrusted) return;
    if (event.key === "Enter" && previewEnterKeys.has(event.code)) {
      if (event.repeat) {
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      // A fresh press also clears a release missed while the page was unfocused.
      previewEnterKeys.delete(event.code);
    }
    if (!document.hasFocus() || document.hidden || event.defaultPrevented || event.repeat ||
        event.isComposing || event.keyCode === 229 || event.ctrlKey || event.metaKey ||
        event.altKey || event.shiftKey) return;
    const key = event.key.toLowerCase();
    if (key !== "t" && !Object.hasOwn(buttonIds, key)) return;
    if (key !== "enter" && !document.querySelector(checkedToggleSelector)) return;
    const focused = [event.target, document.activeElement].filter((node) => node instanceof Element);
    if (focused.some((node) => node.closest(editor) || node.isContentEditable) || otherPopup()) return;
    if (key === "enter" && !focused.every(pageSurface)) return;
    if (preview && !["t", "p", "f"].includes(key)) return;

    if (key === "t") {
      const source = document.querySelector(reference);
      if (!preview && !visible(source)) return;
      if (preview) preview.close();
      else openReference(source);
    } else {
      const button = document.getElementById(buttonIds[key]);
      if (!(button instanceof HTMLButtonElement) || button.disabled || !visible(button)) return;
      button.click();
      if (key === "f" && preview) preview.close();
    }
    event.preventDefault();
    event.stopPropagation();
  }, options);

  window[installation] = () => {
    listeners.abort();
    hints.remove();
    preview?.close();
    return previewEnterKeys;
  };
}
