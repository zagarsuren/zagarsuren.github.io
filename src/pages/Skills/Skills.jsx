import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import IconCloudDemo from "@/components/globe";
import {
  Code2,
  Paintbrush,
  Database,
  Layout,
  Cpu,
  Cloud,
  Handshake,
} from "lucide-react";
import {
  FaReact,
  FaNodeJs,
  FaPython,
  FaDocker,
  FaGitAlt,
  FaLinux,
  FaFigma,
  FaAws,
  FaGoogle,
  FaMicrosoft,
  FaFileExcel,
  FaChartBar
} from "react-icons/fa";
import {
  SiNextdotjs,
  SiTypescript,
  SiTailwindcss,
  SiPostgresql,
  SiMongodb,
  SiGraphql,
  SiJest,
  SiWebpack,
  SiRedux,
  SiFirebase,
  SiVercel,
  SiVite,
  SiNumpy,
  SiPandas,
  SiTensorflow,
  SiPytorch,
  SiScikitlearn,
  SiGooglecloud,
  SiTableau,
  SiOpenai,
  SiOpenaigym,
  SiOpencv,
  SiLangchain
} from "react-icons/si";
import { TbBrandVscode } from "react-icons/tb";
import { BsFileEarmarkCode, BsGrid1X2 } from "react-icons/bs";
import { MdAnimation } from "react-icons/md";
import { FcWorkflow } from "react-icons/fc";

// Swiper
import { Swiper, SwiperSlide } from "swiper/react";
import { Navigation } from "swiper/modules";
import "swiper/css";
import "swiper/css/navigation";

const SkillCard = ({ icon: Icon, title, skills, color }) => (
  <Card className="group relative overflow-hidden bg-gray-900/80 border-gray-700 hover:scale-[1.02] transition-all duration-300 hover:shadow-xl hover:shadow-blue-500/20">
    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-[rgba(100,100,255,0.1)] to-transparent group-hover:via-[rgba(100,100,255,0.2)] animate-shimmer"></div>
    <CardContent className="p-6 relative z-10">
      <div className="flex items-center gap-4 mb-6">
        <div className={`p-3 rounded-xl bg-gray-800/50 ${color} group-hover:scale-110 transition-transform duration-300`}>
          <Icon className="w-8 h-8" />
        </div>
        <h3 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400 font-mono">
          {title}
        </h3>
      </div>
      <div className="flex flex-wrap gap-2">
        {skills.map((skill, index) => (
          <Badge
            key={index}
            variant="outline"
            className="group/badge relative bg-gray-800/50 hover:bg-gray-700/80 text-gray-100 border-gray-600 flex items-center gap-2 py-2 px-3 transition-all duration-300 hover:scale-105 hover:shadow-lg hover:shadow-blue-500/20"
          >
            <span className="transform group-hover/badge:scale-110 transition-transform duration-300">
              {skill.icon}
            </span>
            <span className="font-medium">{skill.name}</span>
          </Badge>
        ))}
      </div>
    </CardContent>
  </Card>
);

const SkillsSection = () => {
  const skillCategories = [
    /*{
      icon: Code2,
      title: "Frontend Development",
      color: "text-blue-400",
      skills: [
        { name: "React", icon: <FaReact className="w-4 h-4 text-[#61DAFB]" /> },
        { name: "Next.js", icon: <SiNextdotjs className="w-4 h-4 text-white" /> },
        { name: "Tailwind CSS", icon: <SiTailwindcss className="w-4 h-4 text-[#38B2AC]" /> },
        { name: "HTML5", icon: <BsFileEarmarkCode className="w-4 h-4 text-[#E34F26]" /> },
        { name: "CSS3", icon: <BsFileEarmarkCode className="w-4 h-4 text-[#1572B6]" /> },
      ],
    },*/
    {
      icon: Database,
      title: "Data Science & Analytics",
      color: "text-pink-400",
      skills: [
        { name: "NumPy", icon: <SiNumpy className="w-4 h-4 text-[#C21325]" /> },
        { name: "Pandas", icon: <SiPandas className="w-4 h-4 text-[#8DD6F9]" /> },
        { name: "Microsoft Excel", icon: <FaFileExcel className="w-4 h-4 text-[#21a366]" /> },
        { name: "Power BI", icon: <FaMicrosoft className="w-4 h-4 text-[#FFCA28]" /> },
        { name: "Tableau", icon: <SiTableau className="w-4 h-4 text-[#E97627]" /> },
        { name: "Scikitlearn", icon: <SiScikitlearn className="w-4 h-4 text-[#FFA800]" /> },
        { name: "SQL", icon: <SiPostgresql className="w-4 h-4 text-[#008bb9]" /> },
        { name: "MongoDB", icon: <SiMongodb className="w-4 h-4 text-[#4DB33D]" /> },
      ],
    },
    {
      icon: Cpu,
      title: "AI, ML & Deep Learning",
      color: "text-pink-400",
      skills: [
        { name: "NumPy", icon: <SiNumpy className="w-4 h-4 text-[#C21325]" /> },
        { name: "Scikitlearn", icon: <SiScikitlearn className="w-4 h-4 text-[#FFA800]" /> },
        { name: "PyTorch", icon: <SiPytorch className="w-4 h-4 text-[#DE3412]" /> },
        { name: "TensorFlow", icon: <SiTensorflow className="w-4 h-4 text-[#FFA800]" /> },
        { name: "OpenAI", icon: <SiOpenai className="w-4 h-4 text-[#99999]" /> },
        { name: "OpenAI Gym", icon: <SiOpenaigym className="w-4 h-4 text-[#99999]" /> },
        { name: "OpenCV", icon: <SiOpencv className="w-4 h-4 text-[#5C3EE3]" /> },
        { name: "ClearML", icon: <SiVercel className="w-4 h-4 text-white" /> },
        { name: "LangChain", icon: <SiLangchain className="w-4 h-4 text-[#4A98A5]" /> },
      ],
    },    
     /*{
      icon: Database,
      title: "Backend Development",
      color: "text-green-400",
      skills: [
        { name: "Node.js", icon: <FaNodeJs className="w-4 h-4 text-[#339933]" /> },
        { name: "Python", icon: <FaPython className="w-4 h-4 text-[#3776AB]" /> },
        { name: "PostgreSQL", icon: <SiPostgresql className="w-4 h-4 text-[#336791]" /> },
        { name: "MongoDB", icon: <SiMongodb className="w-4 h-4 text-[#47A248]" /> },
        { name: "REST APIs", icon: <BsGrid1X2 className="w-4 h-4 text-[#FF6C37]" /> },
      ],
    },
    {
      icon: Layout,
      title: "UI/UX Design",
      color: "text-purple-400",
      skills: [
        { name: "Figma", icon: <FaFigma className="w-4 h-4 text-[#F24E1E]" /> },
        { name: "Responsive Design", icon: <Layout className="w-4 h-4 text-[#38B2AC]" /> },
        { name: "Wireframing", icon: <BsGrid1X2 className="w-4 h-4 text-[#9CA3AF]" /> },
        { name: "Prototyping", icon: <MdAnimation className="w-4 h-4 text-[#F59E0B]" /> },
      ],
    },*/
    {
      icon: Cloud,
      title: "Cloud & DevOps",
      color: "text-orange-400",
      skills: [
        { name: "AWS", icon: <FaAws className="w-4 h-4 text-[#FF9900]" /> },
        { name: "GCP", icon: <SiGooglecloud className="w-4 h-4 text-[#FCC624]" /> },
        { name: "Docker", icon: <FaDocker className="w-4 h-4 text-[#2496ED]" /> },
        { name: "CI/CD", icon: <FcWorkflow className="w-4 h-4" /> },
        /* { name: "Kubernetes", icon: <BsGrid1X2 className="w-4 h-4 text-[#326CE5]" /> },*/
        { name: "Git", icon: <FaGitAlt className="w-4 h-4 text-[#F05032]" /> },
        { name: "Linux", icon: <FaLinux className="w-4 h-4 text-[#FCC624]" /> },
      ],
    },

    /*{
      icon: Handshake,
      title: "Soft Skills",
      color: "text-yellow-400",
      skills: [
        { name: "Problem Solving", icon: <MdAnimation className="w-4 h-4 text-[#FF4081]" /> },
        { name: "Communication", icon: <MdAnimation className="w-4 h-4 text-[#00C853]" /> },
        { name: "Agile/Scrum", icon: <Cpu className="w-4 h-4 text-[#7C4DFF]" /> },
        { name: "Collaboration", icon: <MdAnimation className="w-4 h-4 text-[#FF6D00]" /> },
      ],
    },*/
  ];

  const certificates = [
    {
      title: "Building Generative AI Applications Using Amazon Bedrock",
      issuer: "AWS",
      image: "/certs/aws-bedrock.jpg",
    },
    {
      title: "Databricks Accredited Generative AI Fundamentals",
      issuer: "Databricks",
      image: "/certs/databricks.jpg",
      link: "https://credentials.databricks.com/b5fe450e-9c6e-4587-9bbf-3d54760c6f7e#acc.rOYfxg7G"
    },        
    {
      title: "Data Science Professional",
      issuer: "IBM | Coursera",
      image: "/certs/IBM-DS.png",
      link: "https://www.coursera.org/account/accomplishments/specialization/MMWJ68TWVYTW",
    },
    {
      title: "Machine Learning AWS",
      issuer: "Udacity",
      image: "/certs/udacity.jpg",
      link: "https://s3-us-west-2.amazonaws.com/udacity-printer/production/certificates/0aa1bb89-55d4-4281-b7d6-4551662227e4.pdf"
    },
    {
      title: "Computer Vision",
      issuer: "Kaggle",
      image: "/certs/Kaggle.jpg",
      link: "https://www.kaggle.com/learn/certification/zagarsuren/computer-vision"
    },
  ];

  const badges = [
    {
      title: "Microsoft Cloud & AI Bootcamp 2025",
      issuer: "Microsoft",
      image: "/certs/microsoft-badge.png",
    },
    {
      title: "MongoDB CRUD Operations",
      issuer: "MongoDB",
      image: "/certs/mongodb-badge.png",
      link: "https://www.credly.com/badges/95bb844d-afa0-4cf0-891b-d6e46da4e062"
    },
    {
      title: "Generative AI",
      issuer: "Cognizant | Sydney, Australia",
      image: "/certs/GenAI-badge.png",
    },
    {
      title: "UTS BUILD Global Leadership Program",
      issuer: "University of Technology Sydney",
      image: "/certs/build-bage.png",
      link: "https://au.badgr.com/public/assertions/cJC-NjK_SHi4uZPkjtz1Fw"
    },
    {
      title: "Machine Learning with Python",
      issuer: "IBM",
      image: "/certs/ibm-badge.png",
      link: "https://www.credly.com/earner/earned/badge/cd6f727c-d1ac-4095-b0fe-d3d29053ffd3"
    }
  ];  


const conferences = [
  {
    name: "GitLab Connect Sydney",
    location: "Sydney, Australia",
    date: "July 2025",
  },
  {
    name: "AWS Summit Sydney",
    location: "Sydney, Australia",
    date: "June 2025",
  },
  {
    name: "Australasian Conference on Information Systems (ACIS 2024)",
    location: "Canberra, Australia",
    date: "December 2024",
  },
  {
    name: "Microsoft AI Tour",
    location: "Sydney, Australia",
    date: "December 2024",
  },
  {
    name: "Databricks World Tour Sydney",
    location: "Sydney, Australia",
    date: "Aug 2024",
  },  
  {
    name: "Google I/O Extended",
    location: "Sydney, Australia",
    date: "June 2024",
  },
  {
    name: "Salesforce World Tour",
    location: "Sydney, Australia",
    date: "February 2024",
  },
  {
    name: "UNSW AI Symposium",
    location: "UNSW, Sydney, Australia",
    date: "November 2023",
  },  
  {
    name: "Data Nomads: AI & Data Science Conference",
    location: "Ulaanbaatar, Mongolia",
    date: "April 2020",
  },
  {
    name: "Kaggle Days",
    location: "Beijing, China",
    date: "October 2019",
  }
];

  return (
    <main className="pt-15 lg:pt-0 text-white min-h-screen bg-[#04081A] relative">
      <div className="absolute inset-0 bg-grid-pattern opacity-20 pointer-events-none"></div>

      <section className="container mx-auto px-8 py-11 relative z-10">

        {/* Section header with enhanced effects */}
        <div className="flex flex-col items-center space-y-8 pt-16 mb-10">
          <div className="relative">
            <h2 className="text-3xl md:text-5xl font-black text-transparent bg-gradient-to-r from-teal-400 to-blue-500 bg-clip-text text-center font-mono">
              Skills
            </h2>
            <div className="absolute inset-0 -z-10 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 blur-3xl rounded-full" />
          </div>
          <p className="text-lg md:text-xl text-gray-400 font-medium tracking-wide text-center max-w-2xl font-mono">
            A showcase of my technical expertise and diverse skill set across various domains
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">

          {skillCategories.map((category, index) => (
            <SkillCard
              key={index}
              icon={category.icon}
              title={category.title}
              skills={category.skills}
              color={category.color}
            />
          ))}
        </div>

        {/* Certificates Section */}
        {/* Section header with enhanced effects */}
        <div className="flex flex-col items-center space-y-8 pt-16 mb-10">
          <div className="relative">
            <h2 className="text-3xl md:text-5xl font-black text-transparent bg-gradient-to-r from-teal-400 to-blue-500 bg-clip-text text-center font-mono">
              Certificates
            </h2>
            <div className="absolute inset-0 -z-10 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 blur-3xl rounded-full" />
          </div>
          <p className="text-lg md:text-xl text-gray-400 font-medium tracking-wide text-center max-w-2xl font-mono">
            Certifications that validate my expertise and commitment to continuous learning
          </p>
        </div>

        <Swiper
          modules={[Navigation]}
          navigation
          spaceBetween={20}
          slidesPerView={1}
          breakpoints={{
            640: { slidesPerView: 1 },
            768: { slidesPerView: 2 },
            1024: { slidesPerView: 3 },
          }}
          className="pb-12"
        >
          {certificates.map((cert, index) => (
            <SwiperSlide key={index}>
              <a
                href={cert.link}
                target="_blank"
                rel="noopener noreferrer"
                className="block bg-gray-900/80 border border-gray-700 rounded-xl overflow-hidden shadow-lg transition-transform duration-300 hover:scale-105"
              >
                <img
                  src={cert.image}
                  alt={cert.title}
                  className="w-full h-80 object-cover"
                />
                <div className="p-4">
                  <h3 className="text-xl font-semibold text-white mb-1">
                    {cert.title}
                  </h3>
                  <p className="text-gray-400 text-sm">{cert.issuer}</p>
                </div>
              </a>
            </SwiperSlide>
          ))}
        </Swiper>

        {/* Badges Section */}
        <div className="flex flex-col items-center space-y-8 pt-16 mb-10">
          <div className="relative">
            <h2 className="text-3xl md:text-5xl font-black text-transparent bg-gradient-to-r from-teal-400 to-blue-500 bg-clip-text text-center font-mono">
              Badges
            </h2>
            <div className="absolute inset-0 -z-10 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 blur-3xl rounded-full" />
          </div>
          <p className="text-lg md:text-xl text-gray-400 font-medium tracking-wide text-center max-w-2xl font-mono">
            Digital badges that represent my achievements and skills in various domains
          </p>
        </div>

        <Swiper
          modules={[Navigation]}
          navigation
          spaceBetween={20}
          slidesPerView={1}
          breakpoints={{
            640: { slidesPerView: 2 },
            768: { slidesPerView: 3 },
            1024: { slidesPerView: 4 },
            1280: { slidesPerView: 5 }, // 5 badges visible
          }}
          className="pb-12"
        >
          {badges.map((cert, index) => (
            <SwiperSlide key={index}>
              <a
                href={cert.link}
                target="_blank"
                rel="noopener noreferrer"
                className="block bg-gray-900/80 border border-gray-700 rounded-xl overflow-hidden shadow-lg transition-transform duration-300 hover:scale-105"
              >
                <div className="flex items-center justify-center bg-gray-800 h-44"> 
                  <img
                    src={cert.image}
                    alt={cert.title}
                    className="h-36 object-contain" // fixed height, scales proportionally
                  />
                </div>
                <div className="p-4">
                  <h3 className="text-lg font-semibold text-white mb-1">
                    {cert.title}
                  </h3>
                  <p className="text-gray-400 text-sm">{cert.issuer}</p>
                </div>
              </a>
            </SwiperSlide>
          ))}
        </Swiper>

        {/* Conferences Section */}
        <div className="max-w-6xl mx-auto px-4 mt-12">
          <h2 className="text-3xl md:text-5xl font-black text-transparent bg-gradient-to-r from-teal-400 to-blue-500 bg-clip-text text-center font-mono mb-8">
            Conferences Attended
          </h2>
          
          <div className="overflow-x-auto bg-gray-900/80 border border-gray-700 rounded-xl shadow-lg">
            <table className="min-w-full text-left text-gray-300 font-mono">
              <thead className="bg-gray-800">
                <tr>
                  <th className="px-6 py-3 text-sm font-semibold">Conference Name</th>
                  <th className="px-6 py-3 text-sm font-semibold">Location</th>
                  <th className="px-6 py-3 text-sm font-semibold">Date</th>
                </tr>
              </thead>
              <tbody>
                {conferences.map((conf, index) => (
                  <tr
                    key={index}
                    className="border-t border-gray-700 hover:bg-gray-800/50 transition-colors"
                  >
                    <td className="px-6 py-4">{conf.name}</td>
                    <td className="px-6 py-4">{conf.location}</td>
                    <td className="px-6 py-4">{conf.date}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>


        {/* Section header with enhanced effects */}
        <div className="flex flex-col items-center space-y-8 pt-16 mb-10">
          <div className="relative">
            <h2 className="text-3xl md:text-5xl font-black text-transparent bg-gradient-to-r from-teal-400 to-blue-500 bg-clip-text text-center font-mono">
              Tech Stack
            </h2>
            <div className="absolute inset-0 -z-10 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 blur-3xl rounded-full" />
          </div>
          <p className="text-lg md:text-xl text-gray-400 font-medium tracking-wide text-center max-w-2xl font-mono">
            A collection of tools and technologies that power my projects and solutions
          </p>
        </div>

        <div className="flex justify-center items-center">
          <IconCloudDemo />
        </div>
      </section>

      <style jsx>{`
        @keyframes shimmer {
          0% {
            transform: translateX(-100%);
          }
          100% {
            transform: translateX(100%);
          }
        }
        .animate-shimmer {
          animation: shimmer 2s infinite;
        }
        .bg-grid-pattern {
          background-image: linear-gradient(
              to right,
              rgba(100, 100, 255, 0.1) 1px,
              transparent 1px
            ),
            linear-gradient(
              to bottom,
              rgba(100, 100, 255, 0.1) 1px,
              transparent 1px
            );
          background-size: 30px 30px;
        }
      `}</style>
    </main>
  );
};

export default SkillsSection;
