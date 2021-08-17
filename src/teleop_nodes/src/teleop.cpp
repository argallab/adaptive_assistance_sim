// Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

#include <teleop_nodes/teleop.h>
///////   functions   /////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////

// constructor
TeleopClass::TeleopClass()
{
  nh_.getParam("control_mode", control_mode_);

  //moveit stuff
  robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
  robot_model::RobotModelPtr kinematic_model = robot_model_loader.getModel();
  robot_state::RobotStatePtr kinematic_state(new robot_state::RobotState(kinematic_model));
  robot_state_ptr_ = kinematic_state;
  arm_joint_model_group_ = robot_state_ptr_ -> getJointModelGroup("mico_arm");
  gripper_joint_model_group_ = robot_state_ptr_ -> getJointModelGroup("mico_gripper");
  hand_link_model_ = robot_state_ptr_->getLinkModel("mico_link_hand");
  moveit::core::JointBoundsVector bounds =  arm_joint_model_group_ -> getActiveJointModelsBounds(); //get all the bounds pos, vel, acc size is 6

  jacobian_reference_ << 0.0, 0.0, 0.0;
  update_flag_ = 0;

  dof_ = arm_joint_model_group_ -> getVariableCount();
  num_fingers_ = gripper_joint_model_group_ -> getVariableCount();

  qdiff_lower_.set_size(dof_);
  qdiff_upper_.set_size(dof_);
  dq_target_.set_size(dof_);
  dq_min_current_.set_size(dof_);
  dq_max_current_.set_size(dof_);
  q_max_.set_size(dof_);
  q_min_.set_size(dof_);
  dq_opt_.set_size(dof_);
  ones_.set_size(dof_); //vector containing ones
  zeros_.set_size(dof_); //vector containing zeros

  //bounds
  dq_min_.set_size(dof_);
  dq_max_.set_size(dof_);

  q_.set_size(dof_);
  dq_.set_size(dof_);

  current_twist_6_.set_size(6); //incoming twist needs to be (6+fingers)x1
  current_input_finger_.set_size(num_fingers_);
  jacobian_.set_size(6,dof_);

  //init current_twist with zero
  for (int i = 0; i < 6; i ++)
  {
    current_twist_6_(i) = 0;
  }

  for (int i = 0; i < num_fingers_; i++)
    current_input_finger_(i) = 0;

  //init twist and q and qd, and hardware joint limits
  for (int i = 0; i < dof_; i++)
  {
    dq_opt_ = 0;
    q_(i) = 0;
    dq_(i) = 0;
    dq_min_current_(i) = 0;
    dq_max_current_(i) = 0;
    ones_(i) = 1;
    zeros_(i) = 0;
    // q_min_(i) = (*bounds[i])[0].min_position_;
    // q_max_(i) = (*bounds[i])[0].max_position_;
    q_min_(i) = -10000.0;//bounds only contain -pi / + pi joint angles should be wrapped in order to use these limits
    q_max_(i) = 10000.0;
    // dq_min_(i) = (*bounds[i])[0].min_velocity_;
    // dq_max_(i) = (*bounds[1])[0].max_velocity_;
    dq_min_(i) = -0.85;
    dq_max_(i) = 0.85;
  }

  q_min_(1) = (*bounds[1])[0].min_position_;
  q_min_(2) = (*bounds[2])[0].min_position_;
  q_max_(1) = (*bounds[1])[0].max_position_;
  q_max_(2) = (*bounds[2])[0].max_position_;


  for (int i = 0; i < 10; i++)
  {
    twist_6_filter_list_.push_back(zeros_);
  }


  twist_sub_ = nh_.subscribe("/control_input", 1, &TeleopClass::TwistCB, this);
  joint_state_sub_ = nh_.subscribe("/joint_states", 1, &TeleopClass::JointStateCB, this);

  vel_j0_pub_ = nh_.advertise<std_msgs::Float64>("/mico/arm_0_joint_velocity_controller/command",1);
  vel_j1_pub_ = nh_.advertise<std_msgs::Float64>("/mico/arm_1_joint_velocity_controller/command",1);
  vel_j2_pub_ = nh_.advertise<std_msgs::Float64>("/mico/arm_2_joint_velocity_controller/command",1);
  vel_j3_pub_ = nh_.advertise<std_msgs::Float64>("/mico/arm_3_joint_velocity_controller/command",1);
  vel_j4_pub_ = nh_.advertise<std_msgs::Float64>("/mico/arm_4_joint_velocity_controller/command",1);
  vel_j5_pub_ = nh_.advertise<std_msgs::Float64>("/mico/arm_5_joint_velocity_controller/command",1);
  finger_1_pub_ = nh_.advertise<std_msgs::Float64>("/mico/finger_joint_1_velocity_controller/command",1);
  finger_2_pub_ = nh_.advertise<std_msgs::Float64>("/mico/finger_joint_2_velocity_controller/command",1);

  vel0_.data = 0.0;
  vel1_.data = 0.0;
  vel2_.data = 0.0;
  vel3_.data = 0.0;
  vel4_.data = 0.0;
  vel5_.data = 0.0;
  fin1_.data = 0.0;
  fin2_.data = 0.0;
}


// destructor
TeleopClass::~TeleopClass()
{
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//     CALLBACKS
void TeleopClass::TwistCB(const std_msgs::Float32MultiArray::ConstPtr & new_control_input)
{
  current_control_input_ptr_ = new_control_input;
}


void TeleopClass::JointStateCB(const sensor_msgs::JointState::ConstPtr & new_joint_state)
{
  current_joint_state_ptr_ = new_joint_state;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//   update all the member variables: q_, dq_, jacobian_, twist, q_diff_lower_, q_diff_upper, dq_target_, robot_state)

void TeleopClass::UpdateAll()
{
  if ( ! current_joint_state_ptr_ ||  ! current_control_input_ptr_)
  {
    //come here as long as one pointer is not pointing anywhere
    // bools evalueates to 0 (false) if not pointing anywhere and 1 (true) if pointing to the ros msgs that is received
    update_flag_ = 0;
    tf_listener_.waitForTransform("world", "mico_link_hand", ros::Time(0), ros::Duration(5.0)); //we have to wait anyways for the callbacks to be triggered
    ROS_FATAL("TELEOP: returned before updating variables");
    return;
  }

  update_flag_ = 1;

  UpdateTwist();
  // FilterTwist(); this is now happening in control input

  if (control_mode_ == "joint" && dq_opt_.size() == 6)
  {
    dq_opt_ = current_twist_6_;
    return;
  }

  else if (control_mode_ == "cartesian")
  {
    // ChangeTwistOrientationReference(); this is now happening in control input
    UpdateJointStates();
    dq_opt_ = dq_; //dq_opt_ will be used in the optimization as a reference and contains the optimal joint velocities

    UpdateJacobian(); // update jacobian (includes the update of the robot state according to the current joint values)

    UpdateTargetVelocity();

    UpdateCurrentVelocityLimits();
  }

  else
  {
    ROS_FATAL("param control_mode has to be cartesian or joint, something else is not implemented");
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//updating current joint positions and joint vels

void TeleopClass::UpdateJointStates()
{

  for (int i = 0; i < dof_ ; i++)
  {
    q_(i) = current_joint_state_ptr_ -> position[i];
    dq_(i) = current_joint_state_ptr_ -> velocity[i];
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// find the targeted velocities that take you away from joint positio limits
void TeleopClass::UpdateTargetVelocity()
{
  qdiff_lower_ = DELTA_JOINT_PENALTY * ones_ -  (q_ - q_min_);
  qdiff_upper_ = DELTA_JOINT_PENALTY * ones_  - (q_max_ - q_);

  for (int i = 0; i < dof_; i++)
  {
    if (qdiff_lower_(i) > 0)
    {
      dq_target_(i) = qdiff_lower_(i);
    }
    else if(qdiff_upper_(i) > 0)
    {
      dq_target_(i) = - qdiff_upper_(i);
    }
    else
    {
      dq_target_(i) = 0;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//find the current vel limits
//q_ is the current position, q_min/q_max are joint pos limits , dq_min/max_current are velocity limits according to current pos, dq_min/max are joint vel limits

void TeleopClass::UpdateCurrentVelocityLimits()
{
  for (int i = 0; i < dof_; i++)
  {
    if (q_(i) <= q_min_(i) + JOINT_LIMIT_TOLERANCE)
    {
      dq_min_current_(i) = 0;
      dq_max_current_(i) = dq_max_(i);
    }
    else if(q_(i) >= q_max_(i) - JOINT_LIMIT_TOLERANCE)
    {
      dq_min_current_(i) = dq_min_(i);
      dq_max_current_(i) = 0;
    }
    else
    {
      dq_min_current_(i) = dq_min_(i);
      dq_max_current_(i) = dq_max_(i);
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//  update the twist vector

void TeleopClass::UpdateTwist()
{
  // cartesian twist 6x1
  for (int i = 0; i < 6; i++)
  {
    current_twist_6_(i) = current_control_input_ptr_ -> data[i];
  }

  //update input for fingers that come in the twist msg
  for (int i = 0; i < num_fingers_ ; i++)
  {
    current_input_finger_(i) = current_control_input_ptr_ -> data[6+i];
  }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// smooth twist input

void TeleopClass::FilterTwist()
{
  //smooth the twist input
  twist_6_filter_list_.erase(twist_6_filter_list_.begin());
  twist_6_filter_list_.push_back(current_twist_6_);
  // std::cout << "size of twist list  " << twist_6_filter_list_.size() << std::endl;

  double sum;
  for (int i = 0; i < dof_; i++)
  {
    sum = 0;
    for (int j = 0; j < twist_6_filter_list_.size(); j++)
    {
      // std::cout << "sum: " << sum << " value of twist in list " << (twist_6_filter_list_[j])(i)<< std::endl;
      sum += (twist_6_filter_list_[j])(i)  ;
    }

    current_twist_6_(i) = sum / twist_6_filter_list_.size();
    // std::cout << "mean " << current_twist_6_(i) << std::endl;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// change orientation reference

void TeleopClass::ChangeTwistOrientationReference()
{
  //orientation wrt the end effector
  // the optimization is done wrt to mico_link_hand not mico_end_effector
  tf_listener_.lookupTransform("world", "mico_link_hand", ros::Time(0), tf_); //needs find package tf in cmakelists

  //get transform from world to mico_link_hand in the tf matrix format and convert it to  eigen
  tf::matrixTFToEigen(tf_.getBasis(), Eigen_w_to_eef_);
  //convert from eigen to dlib
  dlib_w_to_eef_ = dlib::mat(Eigen_w_to_eef_);
  //update orientation twist (lower half of twist vector)
  dlib::set_rowm(current_twist_6_, dlib::range(3,5)) =  dlib_w_to_eef_*dlib::rowm(current_twist_6_,dlib::range(3,5));

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//  compute the jacobian

void TeleopClass::UpdateJacobian()
{
  //update robot state
  robot_state_ptr_ -> setVariableValues(*current_joint_state_ptr_);

  robot_state_ptr_ -> getJacobian(arm_joint_model_group_, hand_link_model_, jacobian_reference_, jacobian_Eigen_);


  //convert eigen matrix to dlib matirx
  jacobian_ = dlib::mat(jacobian_Eigen_);
  // ROS_INFO_STREAM("\n\n jacobian: " << jacobian_);
}

void TeleopClass::PublishJointVelocities()
{
      vel0_.data = dq_opt_(0);
      vel1_.data = dq_opt_(1);
      vel2_.data = dq_opt_(2);
      vel3_.data = dq_opt_(3);
      vel4_.data = dq_opt_(4);
      vel5_.data = dq_opt_(5);
      fin1_.data = current_input_finger_(0);
      fin2_.data = current_input_finger_(1);


      vel_j0_pub_.publish(vel0_);
      vel_j1_pub_.publish(vel1_);
      vel_j2_pub_.publish(vel2_);
      vel_j3_pub_.publish(vel3_);
      vel_j4_pub_.publish(vel4_);
      vel_j5_pub_.publish(vel5_);
      finger_1_pub_.publish(fin1_);
      finger_2_pub_.publish(fin2_);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// function classes for optimization

class ObjectiveClass
{
public:
  ObjectiveClass(dlib_matr J, dlib_vec T, dlib_vec dqT):
      jacobian_(J), current_twist_6_(T), dq_target_(dqT)
  {
  }

  double operator () (const dlib_vec & DQ_OPT) const
  {
      double objective;

      dlib_vec delta_twist_;
      delta_twist_.set_size(6);
      delta_twist_ = (jacobian_ * DQ_OPT) - current_twist_6_;
      objective = 0.5 * dlib::trans(delta_twist_)*delta_twist_ + LAMBDA_DQDIST * 0.5 * dlib::trans(DQ_OPT - dq_target_) * (DQ_OPT - dq_target_) ;

      return objective;
  }

  dlib_matr jacobian_;
  dlib_vec current_twist_6_;
  dlib_vec dq_target_;
};


class ObjectiveGradientClass
{
public:
  ObjectiveGradientClass(dlib_matr J, dlib_vec T, dlib_vec dqT):
      jacobian_(J), current_twist_6_(T), dq_target_(dqT)
  {
  }

  const dlib_vec operator () (const dlib_vec & DQ_OPT) const
  {
    dlib_vec gradient_;
    dlib_vec delta_twist_;
    delta_twist_.set_size(6);
    gradient_.set_size(6);

    delta_twist_ = jacobian_ * DQ_OPT - current_twist_6_;
    gradient_ = dlib::trans(jacobian_) * delta_twist_ + LAMBDA_DQDIST * (DQ_OPT - dq_target_);

    return gradient_;
  }


  dlib_matr jacobian_;
  dlib_vec current_twist_6_;
  dlib_vec dq_target_;

};


////////    main    ///////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{

  ros::init(argc, argv, "teleop_node");

  TeleopClass teleop;

  int print_count = 0;

  ros::Rate rate(50);
  while (ros::ok())
  {

    teleop.UpdateAll();
    if(teleop.update_flag_ == 0) // callbacks twist and joint_states don't receive msgs yet
    {
      ROS_FATAL("TELEOP: not ready yet");
      ros::spinOnce();
      rate.sleep();
      continue;
    }

    if (print_count < 1)
    {
      ROS_INFO("TELEOP: all good now ...");
      print_count += 1;
    }


    if (teleop.control_mode_ == "cartesian")
    {
      //only do the optimization if twist input is greater than 0.001, sometimes the optimization does not give 0, even though the twist input is 0
      bool do_opt = false;

      for (int i = 0; i < 6; i++)
      {
        if (fabs(teleop.current_twist_6_(i)) >= 0.001)
        {
          do_opt = true;
          break;
        }
      }

      double res;
      if (do_opt)
      {
         res = dlib::find_min_box_constrained(dlib::lbfgs_search_strategy(40),
                                                    dlib::objective_delta_stop_strategy(1e-10),
                                                    ObjectiveClass(teleop.jacobian_, teleop.current_twist_6_, teleop.dq_target_),
                                                    ObjectiveGradientClass(teleop.jacobian_, teleop.current_twist_6_, teleop.dq_target_),
                                                    teleop.dq_opt_, teleop.dq_min_current_, teleop.dq_max_current_);
      }

      else //set velocities that are sent to the robot to 0 manually
      {
          for (int i = 0; i < teleop.dof_; i ++)
          {
            teleop.dq_opt_(i) = 0.0;
          }
      }
    }



    else if (teleop.control_mode_ == "joint")
    {
      // nothing has to be done because the twist goes directly into joint velocities

      // optionally one can modify the velocities here or in the joy3axis node. in the joy3axis node the velocities are optimized for cartesian teleop
      teleop.dq_opt_(0) *= 1;
      teleop.dq_opt_(1) *= 1;
      teleop.dq_opt_(2) *= 1;
      teleop.dq_opt_(3) *= 1;
      teleop.dq_opt_(4) *= 1;
      teleop.dq_opt_(5) *= 1;
    }


    teleop.PublishJointVelocities();

    ros::spinOnce();
    rate.sleep();
  }

  return 0;
}
