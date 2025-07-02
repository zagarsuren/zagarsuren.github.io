import HeroImg from "@/assets/images/hero.jpg";

export default function About() {
  return (
    <>
      <section id="about" className="py-16 md:py-32  text-white bg-[#04081A]">
        <div className="mx-auto max-w-5xl space-y-8 px-6 md:space-y-16">
          {/*<h2 className="relative z-10 max-w-xl text-4xl font-medium lg:text-5xl text-white">
            Data Scientist, ML Engineer, AI Developer
          </h2>*/}
          <div className="grid gap-6 sm:grid-cols-2 md:gap-12 lg:gap-24">
            <div className="relative mb-6 sm:mb-0">
              <div className="bg-linear-to-b aspect-76/59 relative rounded-2xl p-px from-zinc-300 to-transparent">
                <img
                  src={HeroImg}
                  className="rounded-[15px] shadow block"
                  alt="payments illustration"
                  width={1207}
                  height={929}
                />
              </div>
            </div>

            <div className="relative space-y-4">
              <p className="text-white font-mono">
                👋 Hi, I’m Zagarsuren — an AI developer and researcher with a strong foundation in data science, machine learning, and computer vision. I specialize in building intelligent systems that combine data-driven insights with real-world impact.
              </p>
              
              <p className="text-white font-mono">
                I’m currently pursuing a Master of AI at the University of Technology Sydney, where I’m honing my skills in data analysis, machine learning, and AI development.
              </p>
              {/*<p className="text-white font-mono">
                With hands-on experience in Python, TensorFlow, PyTorch, and open-source frameworks, I’ve worked on projects ranging from disaster damage segmentation using satellite imagery to inclusive AI solutions for accessibility.
              </p>*/}

              <div className="pt-6">
                {/*<blockquote className="border-l-4 border-gray-300 pl-4">
                  <p className="text-white">
                    I'm passionate about bridging technology and social good — and always open to collaborations that challenge the status quo. 
                  </p>

                  <div className="mt-6 space-y-3">
                    <cite className="block font-medium text-white">
                      Zagarsuren Sukhbaatar
                    </cite>
                    
                    <div className="flex items-center gap-2">
                      <img
                        className="h-5 w-fit"
                        src="#" //{OlovaLogo}
                        alt="Olova Logo"
                        height="20"
                        width="auto"
                      />
                      <span className="text-white">DS</span> 
                    </div>
                    
                  </div>
                </blockquote>*/}
              </div>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
