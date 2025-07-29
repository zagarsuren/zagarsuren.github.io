import React, { useState } from "react";
import EducationLoader from "@/components/ui/EducationLoader";
import {
  Star,
  Award,
  Calendar,
  BookOpen,
  Book,
  GraduationCap,
  Building,
  Trophy,
} from "lucide-react";
import { motion } from "framer-motion";


const EducationSection = () => {
  const [hoveredIndex, setHoveredIndex] = useState(null);

  const educationData = [
    {
      degree: "Master of Artificial Intelligence",
      school: "University of Technology Sydney (UTS), Australia",
      department: "School of Computer Science, FEIT",
      logo: "/logos/uts-logo.svg",
      year: "2023-2025",
      achievements: ["Postgraduate Academic Excellence Scholarship", "2024 Dean's List", "2025 Dean's List", "High Distinction Average", "GPA: 6.88/7.0"],
      skills: ["Computer Vision", "Image Processing", "Neural Networks", "Deep Learning", "AI/GenAI", "LLM", "Multimodal AI", "Reinforcement Learning", "Machine Learning", "Data Science", "Software Development", "Agile Methodologies", "Research Methodology",  "Data & AI Ethics"],
      /*description:
        "Focused on core AI subjects with emphasis on practical laboratory work and scientific research methodologies.",*/
    },
    {
      degree: "BUILD Global Leadership Program",
      school: "University of Technology Sydney (UTS), Australia",
      department: "UTS International",
      logo: "/logos/uts-logo.svg",
      year: "2023-2024",
      achievements: ["Points: 100/100", "Certificate of Completion"],
      skills: ["Leadership", "Global citizenship", "Sustainable Development Goals", "Community engagement", "Social Impact", "Intercultural understanding", "Critical thinking"],
      /*description:
        "Focused on core AI subjects with emphasis on practical laboratory work and scientific research methodologies.",*/
    },    
    {
      degree: "Master of Business Administration (MBA)",
      school: "Da-Yeh University, Taiwan",
      department: "Department of Business Administration",
      logo: "/logos/dayeh-logo.png",
      year: "2016-2018",
      achievements: ["GPA: 4.3/4.3"],
      skills: ["Operations Management", "Financial Management", "Marketing Management", "Strategic Management", "Big Data Analysis", "Managerial Economics", "Business Ethics", "Applied Statistics", "Research Methodology", "Project Management"],
      /*description:
        "Completed a comprehensive MBA program focused on core business disciplines including strategy, finance, operations, and marketing. Gained strong analytical and decision-making skills through case studies, group projects, and applied research.",*/
    },    
    {
      degree: "Bachelor of Statistics",
      school: "National University of Mongolia",
      department: "School of Economic Studies",
      logo: "/logos/num-logo.png",
      year: "2008-2012",
      achievements: ["GPA: 3.5/4.0", "Statistics Olympiad Winner"],
      skills: ["Theory of Statistics", "Probability", "Economics", "Econometrics", "Data Analysis", "Statistical Software", "Quantitative Research", "Survey Methodology", "Multivariate Statistics"],
      /*description:
        "Developed strong analytical and critical thinking skills through comprehensive study of statistics.",*/
    }
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
      },
    },
  };

  const cardVariants = {
    hidden: { y: 50, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.6,
        ease: "easeOut",
      },
    },
  };

  return (
    <section className="min-h-screen relative overflow-hidden py-40 bg-[#04081A]">
      {/* Grid Background */}
      <div className="absolute inset-0 z-0">
        <div className="absolute inset-0 bg-grid-white/[0.05] bg-[length:50px_50px]" />
        <div className="absolute inset-0 bg-gradient-to-t from-[#04081A] via-transparent to-[#04081A]" />
        <div className="absolute inset-0 border border-white/[0.05] grid grid-cols-2 md:grid-cols-4" />
      </div>
      
      <div className="max-w-6xl mx-auto px-4 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16 font-mono"
        >
          <h2 className="text-5xl md:text-7xl font-bold bg-gradient-to-r from-teal-400 to-blue-500 bg-clip-text text-transparent mb-6 font-mono">
            Educational Journey
          </h2>
          <p className="text-gray-300 max-w-2xl mx-auto text-lg font-mono">

          </p>
        </motion.div>

        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-1 md:grid-cols-2 gap-8"
        >
          {educationData.map((edu, index) => (
            <motion.div
              key={index}
              variants={cardVariants}
              className={`relative border rounded-xl p-8 transition-all duration-300 bg-gray-900/50 backdrop-blur-sm ${
                hoveredIndex === index
                  ? "border-teal-500 scale-[1.02]"
                  : "border-blue-400/20"
              }`}
              onMouseEnter={() => setHoveredIndex(index)}
              onMouseLeave={() => setHoveredIndex(null)}
            >
              <div className="space-y-6">
                <div className="space-y-2">
                  <div className="flex items-center gap-3">
                    <img src={edu.logo} alt="UTS Logo" className="w-10 h-10 object-contain" />
                    <h3 className="text-2xl font-bold text-white font-mono">
                      {edu.degree}
                    </h3>
                  </div>
                  <p className="text-lg text-gray-300 flex items-center gap-2">
                    <Building className="w-5 h-5 text-teal-500" />
                    {edu.school}
                  </p>
                  <p className="text-lg text-gray-300 flex items-center gap-2">
                    <BookOpen className="w-5 h-5 text-teal-500" />
                    {edu.department}
                  </p>
                  <p className="text-gray-400 flex items-center gap-2 font-mono">
                    <Calendar className="w-4 h-4" />
                    {edu.year}
                  </p>
                </div>

                <p className="text-gray-300 text-sm italic border-l-2 border-teal-500 pl-3 font-mono">
                  {edu.description}
                </p>

                <div className="space-y-3">
                  <h4 className="text-sm font-semibold text-white flex items-center gap-2 font-mono">
                    <Trophy className="w-4 h-4 text-yellow-500" />
                    Key Achievements
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {edu.achievements.map((achievement, i) => (
                      <div
                        key={i}
                        className="px-3 py-1 rounded-full bg-teal-500/10 text-teal-400 flex items-center gap-2 text-sm font-mono"
                      >
                        <Award className="w-4 h-4" />
                        <span>{achievement}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  {edu.skills.map((skill, i) => (
                    <span
                      key={i}
                      className="px-2 py-1 text-xs rounded bg-blue-500/10 text-blue-300 font-mono"
                    >
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

export default EducationSection;
