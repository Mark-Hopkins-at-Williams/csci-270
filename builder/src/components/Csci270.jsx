import courseData from "./csci270.json";

const Welcome = () => {
  return (
    <div
      className="welcome"
      style={{
        display: "flex",
        flexFlow: "row wrap",
        justifyContent: "center",
      }}
    >
      <div
        style={{
          alignContent: "center",
        }}
      >
        <div
          className="csci270-welcome-logo"
          style={{ color: "#41FF00", fontSize: "100px" }}
        >
          270
        </div>
        <div
          className="csci270-welcome"
          style={{ fontSize: "20px", color: "#aaFF44" }}
        >
          foundations of
        </div>
        <div
          className="csci270-welcome"
          style={{ fontSize: "20px", color: "#aaFF44" }}
        >
          artificial intelligence
        </div>
      </div>
    </div>
  );
};

const TeachingAssistant = ({ image, name, hours, hours2, where }) => {
  return (
    <div
      className="csci270-subtitle"
      style={{
        display: "flex",
        flexFlow: "column nowrap",
        justifyContent: "stretch",
        fontSize: "14px",
      }}
    >
      <img
        src={image}
        style={{
          borderStyle: "solid",
          borderColor: "white",
          width: "100px",
        }}
      />
      <div
        style={{
          width: "100px",
          fontWeight: "bold",
          fontSize: "29px",
        }}
      >
        {name.toLowerCase()}
      </div>
      <div>{hours.toLowerCase()}</div>
      {hours2 ? <div>{hours2.toLowerCase()}</div> : <div></div>}
      <div>{where.toLowerCase()}</div>
    </div>
  );
};

const TeachingAssistants = () => {
  return (
    <div className="teaching-assistants">
      <div
        style={{
          display: "flex",
          flexFlow: "column nowrap",
          justifyContent: "stretch",
          height: "100%",
        }}
      >
        <div style={{ flexGrow: 1, flexShrink: 1 }}></div>
        <div
          style={{
            display: "flex",
            flexFlow: "row wrap",
            justifyContent: "space-evenly",
          }}
        >
          <TeachingAssistant
            image="images/mark.png"
            name="Mark"
            hours="tu, th 11am-noon"
            where="tcl 307"
          />
          <TeachingAssistant
            image="images/mendez.jpeg"
            name="Juan"
            hours="wed 2-4pm"
            where="tcl 307"
          />
          <TeachingAssistant
            image="images/hall.jpeg"
            name="Lauren"
            hours="wed 2-4pm"
            where="tcl 307"
          />
          <TeachingAssistant
            image="images/wexler.jpeg"
            name="Sam"
            hours="wed 2-4pm"
            where="tcl 307"
          />
          <TeachingAssistant
            image="images/park.jpeg"
            name="Serah"
            hours="wed 2-4pm"
            where="tcl 307"
          />
          <TeachingAssistant
            image="images/yuri.jpeg"
            name="Yuri"
            hours="wed 2-4pm"
            where="tcl 307"
          />
        </div>
        <div style={{ flexGrow: 1, flexShrink: 1 }}></div>
      </div>
    </div>
  );
};

const WeekItem = ({ title }) => {
  const renderContent = () => {
    return (
      <div
        className={"csci270-text csci270-locked"}
        style={{
          borderStyle: "solid 2px black",
          margin: "2px",
          padding: "2px",
          color: "white",
          alignContent: "center",
        }}
      >
        <span>{title}</span>
      </div>
    );
  };

  return renderContent();
};

const LabAssignment = ({ type, title, link, release }) => {
  const now = new Date(Date.now());
  const released = release ? now > new Date(release) : false;
  const locked = !released || !link;

  const renderContent = () => {
    return (
      <div
        className={
          locked
            ? "csci270-lab-text csci270-locked"
            : "csci270-lab-text csci270-lab"
        }
        style={{
          border: "5px solid rgb(40, 40, 50)",
          margin: "0",
          padding: "2px",
          width: "100%",
          height: "100%",
          boxSizing: "border-box",
          alignContent: "center",
        }}
      >
        <span>{title}</span>
      </div>
    );
  };

  return !locked ? (
    <a href={link} target="_blank">
      {renderContent()}
    </a>
  ) : (
    renderContent()
  );
};

const SlideDeck = ({ type, title, link, release }) => {
  const now = new Date(Date.now());
  const released = release ? now > new Date(release) : false;
  const locked = !released || !link;
  const iconMap = {
    lecture: "👨‍🏫",
    activity: "✍️",
    kahoot: "🏆",
    holiday: "🏖️",
    movie: "🍿",
    workshop: "🔧",
    quiz: "💪",
  };

  const icon = iconMap[type] ?? "";

  const renderContent = () => {
    return (
      <div
        className={
          locked ? "csci270-text csci270-locked" : "csci270-text csci270-button"
        }
        style={{
          border: "solid 2px rgb(40, 40, 50)",
          margin: "2px",
          padding: "2px",
        }}
      >
        <span>
          {icon} {title}
        </span>{" "}
      </div>
    );
  };

  return !locked ? (
    <a href={link} target="_blank">
      {renderContent()}
    </a>
  ) : (
    renderContent()
  );
};

const Schedule = () => {
  const columnProportions = "1fr 2fr 2fr 2fr";
  const startMonday = new Date(2026, 1, 9); // Feb is month 1 (0-based!)

  const weeks = [];

  for (let week = 1; week <= 20; week++) {
    const monday = new Date(startMonday);
    monday.setDate(startMonday.getDate() + (week - 1) * 7);

    const friday = new Date(monday);
    friday.setDate(monday.getDate() + 4);

    weeks.push({
      week,
      monday,
      friday,
    });
  }

  return (
    <div className="csci270-colorblock">
      <div
        style={{
          display: "flex",
          flexFlow: "column",
          justifyContent: "stretch",
          height: "100%",
        }}
      >
        <div
          style={{
            display: "grid",
            gridTemplateColumns: columnProportions,
          }}
        >
          <div className="csci270-schedule-header">week</div>
          <div className="csci270-schedule-header">mon</div>
          <div className="csci270-schedule-header">wed</div>
          <div className="csci270-schedule-header">thu</div>
        </div>
        <div
          style={{
            display: "flex",
            flexFlow: "column nowrap",
            justifyContent: "stretch",
            height: "100%",
          }}
        >
          {courseData.schedule.map((week) => (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: columnProportions,
              }}
            >
              <WeekItem
                title={`${weeks[week.week - 1].monday.toDateString().toLowerCase()} – ${weeks[week.week - 1].friday.toDateString().toLowerCase()}`}
              />

              <div
                style={{
                  display: "flex",
                  flexFlow: "column nowrap",
                  justifyContent: "stretch",
                  height: "100%",
                }}
              >
                {week.mon.map((content) => (
                  <SlideDeck
                    type={content.type}
                    title={content.title}
                    link={content.link}
                    release={content.release}
                  />
                ))}
              </div>
              <LabAssignment
                type={week.lab.type}
                title={week.lab.title}
                link={week.lab.link}
                release={week.lab.release}
              />
              <div
                style={{
                  display: "flex",
                  flexFlow: "column nowrap",
                  justifyContent: "stretch",
                  height: "100%",
                }}
              >
                {week.thu.map((content) => (
                  <SlideDeck
                    type={content.type}
                    title={content.title}
                    link={content.link}
                    release={content.release}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const Information = () => {
  return (
    <div
      style={{
        display: "flex",
        flexFlow: "row wrap",
        justifyContent: "space-around",
        gap: "20px",
        height: "auto",
        padding: "10px",
        borderStyle: "solid",
      }}
    >
      {courseData.info.map((content) => (
        <Info title={content.title} link={content.link} />
      ))}
    </div>
  );
};

const Info = ({ title, link }) => {
  const renderContent = () => {
    return (
      <div
        className={"csci270-info"}
        style={{
          margin: "2px",
          padding: "2px",
        }}
      >
        <span>{title}</span>{" "}
      </div>
    );
  };

  return (
    <a href={link} target="_blank">
      {renderContent()}
    </a>
  );
};

function csci270() {
  return (
    <div>
      <div
        className="csci270"
        style={{
          display: "flex",
          flexFlow: "column nowrap",
          justifyContent: "center",
          alignItems: "stretch",
          gap: "20px",
          height: "auto",
          paddingBottom: "20px",
        }}
      >
        <Welcome />
        <TeachingAssistants />
        <Information />
        <div
          style={{
            flexGrow: 1,
            display: "flex",
            flexFlow: "column nowrap",
            justifyContent: "stretch",
            gap: "20px",
            height: "auto",
          }}
        >
          <div
            style={{
              display: "flex",
              flexFlow: "row nowrap",
              flexGrow: 1,
              flexShrink: 1,
              justifyContent: "stretch",
              gap: "20px",
            }}
          >
            <Schedule />
          </div>
        </div>
      </div>
    </div>
  );
}

export default csci270;
