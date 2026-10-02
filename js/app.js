import { makeZip } from "./zip.js";
import { RAW_EXTENSIONS, isRaw } from "./formats.js";

const drop = document.getElementById("drop");
const picker = document.getElementById("picker");
const mergeBtn = document.getElementById("merge");
const download = document.getElementById("download");
const status = document.getElementById("status");
const DROP_HINT = drop.textContent;
picker.accept = RAW_EXTENSIONS.flatMap((e) => [e, e.toUpperCase()]).join(",");
let files = [];

function setStatus(text, error = false) {
  status.textContent = text;
  status.className = error ? "error" : "";
}

function setFiles(list) {
  files = [...list].filter((f) => isRaw(f.name)).sort((a, b) => a.name.localeCompare(b.name));
  drop.textContent = files.length ? `${files.length} plików RAW: ${files.map((f) => f.name).join(", ")}` : DROP_HINT;
  mergeBtn.disabled = !files.length;
  download.hidden = true;
  setStatus("");
}

drop.onclick = () => picker.click();
picker.onchange = () => setFiles(picker.files);
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); setFiles(e.dataTransfer.files); };

mergeBtn.onclick = async () => {
  const n = Number(document.getElementById("group").value);
  if (!(n >= 1) || files.length % n) {
    setStatus(`Błąd: liczba plików RAW (${files.length}) nie dzieli się przez ${n}.`, true);
    return;
  }
  if (!self.crossOriginIsolated) {
    setStatus("Błąd: strona nie jest izolowana (brak SharedArrayBuffer). Odśwież stronę.", true);
    return;
  }
  mergeBtn.disabled = true;
  download.hidden = true;
  setStatus("Wczytywanie plików…");

  const groups = [];
  const transfer = [];
  for (let i = 0; i < files.length; i += n) {
    const group = [];
    for (const f of files.slice(i, i + n)) {
      const buffer = await f.arrayBuffer();
      group.push({ name: f.name, buffer });
      transfer.push(buffer);
    }
    groups.push(group);
  }

  const results = [];
  const warnings = [];
  const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  try {
    await new Promise((resolve, reject) => {
      worker.onmessage = ({ data }) => {
        if (data.type === "progress") setStatus(`${data.text}…`);
        else if (data.type === "result") { results.push(data); warnings.push(...data.warnings); }
        else if (data.type === "done") resolve();
        else if (data.type === "error") reject(new Error(data.message));
      };
      worker.onerror = (e) => reject(new Error(e.message || "błąd workera"));
      worker.postMessage({
        groups,
        opts: {
          lensCorrection: document.getElementById("lens").checked,
          autoWb: document.getElementById("awb").checked,
        },
      }, transfer);
    });

    const out = results.length === 1
      ? { name: results[0].name, blob: results[0].blob }
      : { name: "merged.zip", blob: await makeZip(results) };
    if (download.href) URL.revokeObjectURL(download.href);
    download.href = URL.createObjectURL(out.blob);
    download.download = out.name;
    download.hidden = false;
    setStatus([`Gotowe: ${out.name}`, ...warnings].join("\n"));
  } catch (err) {
    setStatus(`Błąd: ${err.message}`, true);
  } finally {
    worker.terminate(); // zwalnia pamięć (kilka GB przy pełnej rozdzielczości)
    mergeBtn.disabled = false;
  }
};
