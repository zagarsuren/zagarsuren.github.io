import IconCloud from "./ui/icon-cloud";

const techGroups = [
  {
    category: "Machine Learning & AI",
    slugs: [
      "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
      "jupyter", "openai", "kaggle", "huggingface", "langchain", "fastapi"
    ]
  },
  {
    category: "Cloud & DevOps",
    slugs: ["amazonaws", "googlecloud", "docker", "git", "linux"]
  },
  {
    category: "Frontend & Tools",
    slugs: ["javascript", "react", "visualstudiocode", "figma"]
  },
  {
    category: "Backend & APIs",
    slugs: [
      "python", "flask", "django", "streamlit", "fastapi", "postgresql", "mongodb"
    ]
  },
  {
    category: "Data Visualization & BI",
    slugs: ["tableau", "powerbi"]
  },
  {
    category: "Computer Vision",
    slugs: ["opencv"]
  },
  {
    category: "General Utilities",
    slugs: ["github"]
  }
];

const slugs = techGroups.flatMap(group => group.slugs);

function IconCloudDemo() {
  return (
    <div className="relative flex size-full max-w-6xl items-center justify-center overflow-hidden rounded-lg px-6 pb-10 pt-8 bg-transparent">
      <IconCloud iconSlugs={slugs} size={72} />
    </div>
  );
}

export default IconCloudDemo;
