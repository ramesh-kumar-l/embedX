import * as vscode from "vscode";
import { EmbedXClient } from "./api";

function getClient(): EmbedXClient {
  const cfg = vscode.workspace.getConfiguration("embedx");
  const url = cfg.get<string>("serverUrl", "http://127.0.0.1:8000");
  return new EmbedXClient(url);
}

function getTopK(): number {
  return vscode.workspace.getConfiguration("embedx").get<number>("defaultTopK", 5);
}

// ── EmbedX: Search ──────────────────────────────────────────────────────────

async function cmdSearch(): Promise<void> {
  const query = await vscode.window.showInputBox({
    prompt: "EmbedX: Enter your search query",
    placeHolder: "e.g. how does caching reduce costs?",
  });
  if (!query) return;

  try {
    const results = await getClient().search(query, getTopK());
    if (results.length === 0) {
      vscode.window.showInformationMessage("EmbedX: No results found.");
      return;
    }
    const items = results.map((r, i) => ({
      label: `$(list-ordered) ${i + 1}.  [${r.score.toFixed(3)}]  ${r.text.slice(0, 80)}`,
      detail: r.text,
      description: r.metadata ? JSON.stringify(r.metadata) : "",
    }));
    const picked = await vscode.window.showQuickPick(items, {
      placeHolder: `Top ${results.length} results for: "${query}"`,
      matchOnDetail: true,
    });
    if (picked) {
      await vscode.env.clipboard.writeText(picked.detail ?? "");
      vscode.window.showInformationMessage("EmbedX: Copied to clipboard.");
    }
  } catch (err) {
    vscode.window.showErrorMessage(`EmbedX: ${(err as Error).message}`);
  }
}

// ── EmbedX: Add Selection ───────────────────────────────────────────────────

async function cmdAddSelection(): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("EmbedX: No active editor.");
    return;
  }
  const text = editor.document.getText(editor.selection).trim();
  if (!text) {
    vscode.window.showWarningMessage("EmbedX: No text selected.");
    return;
  }
  try {
    const result = await getClient().add(text);
    const icons: Record<string, string> = {
      added: "$(add)",
      exact_hit: "$(check)",
      semantic_hit: "$(symbol-misc)",
    };
    const icon = icons[result.status] ?? "$(info)";
    vscode.window.showInformationMessage(
      `EmbedX ${icon} ${result.status}  id=${result.id}  (${result.latency_ms.toFixed(1)}ms)`
    );
  } catch (err) {
    vscode.window.showErrorMessage(`EmbedX: ${(err as Error).message}`);
  }
}

// ── EmbedX: Show Stats ──────────────────────────────────────────────────────

async function cmdShowStats(): Promise<void> {
  try {
    const s = await getClient().stats();
    const lines = [
      `Documents     : ${s.document_count}`,
      `Total requests: ${s.total_requests}`,
      `Cache hit rate: ${(s.hit_rate * 100).toFixed(1)}%`,
      `  Exact hits  : ${s.exact_hits}`,
      `  Semantic    : ${s.semantic_hits}`,
      `Cost saved    : $${s.cost_saved_usd.toFixed(6)}`,
      `Tokens saved  : ${s.tokens_saved.toLocaleString()}`,
      `Avg latency   : ${s.latency?.avg_ms?.toFixed(1) ?? "?"}ms`,
    ];
    await vscode.window.showInformationMessage(lines.join("\n"), { modal: true }, "OK");
  } catch (err) {
    vscode.window.showErrorMessage(`EmbedX: ${(err as Error).message}`);
  }
}

// ── Activation ───────────────────────────────────────────────────────────────

export function activate(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("embedx.search", cmdSearch),
    vscode.commands.registerCommand("embedx.addSelection", cmdAddSelection),
    vscode.commands.registerCommand("embedx.showStats", cmdShowStats)
  );
}

export function deactivate(): void {}
