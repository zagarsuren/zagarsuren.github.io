import { useParams, Link } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import { useEffect, useState } from "react";
import { ArrowLeft } from "lucide-react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { atomOneDark } from "react-syntax-highlighter/dist/esm/styles/hljs";

// register your language(s)
import python from "react-syntax-highlighter/dist/esm/languages/hljs/python";
SyntaxHighlighter.registerLanguage("python", python);

// ✂️ Remove the theme’s bg by cloning & zeroing it out
const transparentAtomOneDark = {
  ...atomOneDark,
  hljs: {
    ...atomOneDark.hljs,
    background: "transparent",
  },
};

export default function ProjectDetails() {
  const { id } = useParams();
  const [content, setContent] = useState("");

  useEffect(() => {
    fetch(`/projects/project-${id}.md`)
      .then((res) => res.text())
      .then((text) => setContent(text))
      .catch(() => setContent("# 404\n\nProject not found."));
  }, [id]);

  return (
    <div className="min-h-screen bg-black text-white pt-28 pb-12 px-4">
      <div className="max-w-3xl mx-auto">
        <Link
          to="/projects"
          className="inline-flex items-center gap-2 text-teal-400 hover:text-teal-200 mb-8"
        >
          <ArrowLeft size={16} />
          Back to Projects
        </Link>
        <article className="prose prose-invert prose-lg max-w-none">
          <ReactMarkdown
            children={content}
            components={{
              code({ node, inline, className, children, ...props }) {
                const match = /language-(\w+)/.exec(className || "");
                // only syntax-highlight non-inline code blocks
                if (!inline && match) {
                  return (
                    <div className="my-6 overflow-x-auto rounded-lg bg-zinc-900 text-sm">
                      <SyntaxHighlighter
                        language={match[1]}
                        style={transparentAtomOneDark}        // use transparent theme
                        showLineNumbers={false}
                        wrapLines={false}
                        wrapLongLines={false}
                        PreTag="div"
                        customStyle={{
                          backgroundColor: "transparent",     // ensure no bg
                          padding: "1.0rem",
                          fontSize: "1.0rem",
                          lineHeight: "1.5",
                          borderRadius: "0.5rem",
                          overflowX: "auto",
                        }}
                        codeTagProps={{
                          style: { backgroundColor: "transparent" },
                        }}
                        lineProps={{
                          style: {
                            backgroundColor: "transparent",   // no per-line bg
                            borderLeft: "none",                // no line gutter
                          },
                        }}
                        {...props}
                      >
                        {String(children).replace(/\n$/, "")}
                      </SyntaxHighlighter>
                    </div>
                  );
                }
                // fallback for inline code
                return (
                  <code className="bg-zinc-800 text-pink-300 px-1 py-0.5 rounded">
                    {children}
                  </code>
                );
              },
            }}
          />
        </article>
      </div>
    </div>
  );
}
