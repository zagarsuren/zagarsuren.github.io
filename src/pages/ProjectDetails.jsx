import { useParams, Link } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import { useEffect, useState } from "react";
import { ArrowLeft } from "lucide-react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { atomOneDark } from "react-syntax-highlighter/dist/esm/styles/hljs";

// Optional: register languages (Python in your case)
import python from "react-syntax-highlighter/dist/esm/languages/hljs/python";
SyntaxHighlighter.registerLanguage("python", python);

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
                return !inline && match ? (
                  <div className="my-6 overflow-x-auto rounded-lg bg-zinc-900 text-sm">
                    <SyntaxHighlighter
                      language={match[1]}
                      style={atomOneDark}
                      showLineNumbers={false}
                      wrapLines={false}
                      wrapLongLines={false}
                      customStyle={{
                        background: "transparent",
                        padding: "1.25rem",
                        fontSize: "0.9rem",
                        lineHeight: "1.6",
                        borderRadius: "0.5rem",
                        overflowX: "auto",
                      }}
                      lineProps={(lineNumber) => ({
                        style: {
                          borderLeft: "none", // 🔥 this removes the vertical line
                          backgroundColor: "transparent", // ensures no background highlight
                        },
                      })}
                      PreTag="div"
                      {...props}
                    >
                      {String(children).replace(/\n$/, "")}
                    </SyntaxHighlighter>
                  </div>
                ) : (
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
