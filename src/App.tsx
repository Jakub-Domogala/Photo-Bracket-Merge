import * as React from "react"
import { Download, ImagePlus, Loader2 } from "lucide-react"

import { SupportFooter } from "@/components/support-footer"
import { ThemeToggle } from "@/components/theme-toggle"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import {
  loadFormats,
  loadZip,
  WORKER_URL,
  type RawFile,
} from "@/lib/engine"
import { cn } from "@/lib/utils"

type Download = { url: string; name: string }
type Formats = Awaited<ReturnType<typeof loadFormats>>

const DROP_HINT =
  "Upuść tu pliki RAW (ARW, CR2, CR3, NEF, RAF, DNG, ORF, RW2, …) albo kliknij, żeby wybrać"

export function App() {
  const picker = React.useRef<HTMLInputElement>(null)
  const [formats, setFormats] = React.useState<Formats | null>(null)
  const [files, setFiles] = React.useState<File[]>([])
  const [dragOver, setDragOver] = React.useState(false)
  const [group, setGroup] = React.useState("3")
  const [lens, setLens] = React.useState(true)
  const [awb, setAwb] = React.useState(false)
  const [tiff, setTiff] = React.useState(false)
  const [busy, setBusy] = React.useState(false)
  const [status, setStatus] = React.useState("")
  const [isError, setIsError] = React.useState(false)
  const [jpegDl, setJpegDl] = React.useState<Download | null>(null)
  const [tiffDl, setTiffDl] = React.useState<Download | null>(null)

  React.useEffect(() => {
    loadFormats().then(setFormats)
  }, [])

  React.useEffect(
    () => () => {
      if (jpegDl) URL.revokeObjectURL(jpegDl.url)
      if (tiffDl) URL.revokeObjectURL(tiffDl.url)
    },
    [jpegDl, tiffDl]
  )

  function report(text: string, error = false) {
    setStatus(text)
    setIsError(error)
  }

  function chooseFiles(list: FileList | null) {
    if (!formats || !list) return
    const raw = [...list]
      .filter((f) => formats.isRaw(f.name))
      .sort((a, b) => a.name.localeCompare(b.name))
    setFiles(raw)
    setJpegDl(null)
    setTiffDl(null)
    report("")
  }

  async function merge() {
    const n = Number(group)
    if (!(n >= 1) || files.length % n) {
      report(
        `Błąd: liczba plików RAW (${files.length}) nie dzieli się przez ${group}.`,
        true
      )
      return
    }
    if (!self.crossOriginIsolated) {
      report(
        "Błąd: strona nie jest izolowana (brak SharedArrayBuffer). Odśwież stronę.",
        true
      )
      return
    }
    setBusy(true)
    setJpegDl(null)
    setTiffDl(null)
    report("Wczytywanie plików…")

    const groups: { name: string; buffer: ArrayBuffer }[][] = []
    const transfer: ArrayBuffer[] = []
    for (let i = 0; i < files.length; i += n) {
      const g = []
      for (const f of files.slice(i, i + n)) {
        const buffer = await f.arrayBuffer()
        g.push({ name: f.name, buffer })
        transfer.push(buffer)
      }
      groups.push(g)
    }

    type Result = {
      name: string
      blob: Blob
      tiff?: Blob
      tiffName?: string
      warnings: string[]
    }
    const results: Result[] = []
    const warnings: string[] = []
    const worker = new Worker(WORKER_URL, { type: "module" })
    try {
      await new Promise<void>((resolve, reject) => {
        worker.onmessage = ({ data }) => {
          if (data.type === "progress") report(`${data.text}…`)
          else if (data.type === "result") {
            results.push(data)
            warnings.push(...data.warnings)
          } else if (data.type === "done") resolve()
          else if (data.type === "error") reject(new Error(data.message))
        }
        worker.onerror = (e) => reject(new Error(e.message || "błąd workera"))
        worker.postMessage(
          {
            groups,
            opts: {
              lensCorrection: lens,
              autoWb: awb,
              tiff,
              // ?cpu wyłącza WebGPU (porównania)
              gpu: !new URLSearchParams(location.search).has("cpu"),
            },
          },
          transfer
        )
      })

      // jeden bracket -> sam plik, kilka -> ZIP (osobny na JPEG-i i na TIFF-y)
      const { makeZip } = await loadZip()
      const offer = async (list: RawFile[], zipName: string) => {
        const out =
          list.length === 1
            ? list[0]
            : { name: zipName, blob: await makeZip(list) }
        return { url: URL.createObjectURL(out.blob), name: out.name }
      }
      const jpeg = await offer(
        results.map((r) => ({ name: r.name, blob: r.blob })),
        "merged.zip"
      )
      setJpegDl(jpeg)
      const names = [jpeg.name]
      if (results.every((r) => r.tiff)) {
        const t = await offer(
          results.map((r) => ({ name: r.tiffName!, blob: r.tiff! })),
          "merged_tiff.zip"
        )
        setTiffDl(t)
        names.push(t.name)
      }
      report([`Gotowe: ${names.join(", ")}`, ...warnings].join("\n"))
    } catch (err) {
      report(`Błąd: ${(err as Error).message}`, true)
    } finally {
      worker.terminate() // zwalnia pamięć (kilka GB przy pełnej rozdzielczości)
      setBusy(false)
    }
  }

  const accept = formats?.RAW_EXTENSIONS.flatMap((e) => [
    e,
    e.toUpperCase(),
  ]).join(",")

  return (
    <div className="flex min-h-svh flex-col">
      <main className="mx-auto flex w-full max-w-xl flex-1 flex-col gap-4 p-4">
        <header className="flex items-start justify-between gap-4 pt-2">
          <div>
            <h1 className="font-heading text-xl font-semibold">
              Photo Bracket Merge
            </h1>
            <p className="text-sm text-muted-foreground">
              Wszystko liczy się w przeglądarce — zdjęcia nigdzie nie są
              wysyłane.
            </p>
          </div>
          <ThemeToggle />
        </header>

        <button
          type="button"
          onClick={() => picker.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            chooseFiles(e.dataTransfer.files)
          }}
          className={cn(
            "flex flex-col items-center gap-2 rounded-xl border-2 border-dashed px-4 py-10 text-center text-sm [overflow-wrap:anywhere] text-muted-foreground transition-colors hover:bg-muted/50",
            dragOver && "border-primary bg-muted"
          )}
        >
          <ImagePlus className="size-6" />
          {files.length
            ? `${files.length} plików RAW: ${files.map((f) => f.name).join(", ")}`
            : DROP_HINT}
        </button>
        <input
          ref={picker}
          type="file"
          multiple
          hidden
          accept={accept}
          onChange={(e) => chooseFiles(e.target.files)}
        />

        <Card>
          <CardHeader>
            <CardTitle>Opcje</CardTitle>
            <CardDescription>
              Domyślne ustawienia działają bez ręcznej obróbki.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="flex items-center justify-between gap-4">
              <Label htmlFor="group">Zdjęć w jednym bracketcie</Label>
              <Input
                id="group"
                type="number"
                min={1}
                value={group}
                onChange={(e) => setGroup(e.target.value)}
                className="w-20"
              />
            </div>
            <div className="flex items-center justify-between gap-4">
              <Label htmlFor="lens">
                Korekcja obiektywu (dystorsja i winieta jak w JPEG-u z aparatu)
              </Label>
              <Switch id="lens" checked={lens} onCheckedChange={setLens} />
            </div>
            <div className="flex items-center justify-between gap-4">
              <Label htmlFor="awb">Auto balans bieli</Label>
              <Switch id="awb" checked={awb} onCheckedChange={setAwb} />
            </div>
            <div className="flex items-center justify-between gap-4">
              <Label htmlFor="tiff">
                Przygotuj też TIFF 16-bit do dalszej obróbki (ok. 150 MB na
                zdjęcie 24 Mpx)
              </Label>
              <Switch id="tiff" checked={tiff} onCheckedChange={setTiff} />
            </div>
          </CardContent>
        </Card>

        <div className="flex flex-wrap gap-2">
          <Button
            onClick={merge}
            disabled={busy || !files.length || !formats}
            size="lg"
          >
            {busy && <Loader2 className="animate-spin" />}
            Połącz
          </Button>
          {jpegDl && (
            <Button
              variant="secondary"
              size="lg"
              render={<a href={jpegDl.url} download={jpegDl.name} />}
            >
              <Download />
              Pobierz
            </Button>
          )}
          {tiffDl && (
            <Button
              variant="secondary"
              size="lg"
              render={<a href={tiffDl.url} download={tiffDl.name} />}
            >
              <Download />
              Pobierz TIFF 16-bit
            </Button>
          )}
        </div>

        <p
          role="status"
          className={cn(
            "min-h-5 text-sm whitespace-pre-line",
            isError && "text-destructive"
          )}
        >
          {status}
        </p>
      </main>
      <SupportFooter />
    </div>
  )
}

export default App
