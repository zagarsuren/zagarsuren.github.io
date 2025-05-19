import { useParams, Link } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import { useEffect, useState } from "react";
import { ArrowLeft } from "lucide-react";

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
        {/* Back to Projects Link */}
        <Link
          to="/"
          className="inline-flex items-center gap-2 text-teal-400 hover:text-teal-200 mb-8"
        >
          <ArrowLeft size={16} />
          Back
        </Link>

        {/* Markdown-rendered content */}
        <article className="prose prose-invert prose-lg max-w-none">
          <ReactMarkdown>{content}</ReactMarkdown>
        </article>
      </div>
    </div>
  );
}
